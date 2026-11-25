#!/usr/bin/env python3
"""
Parse influential papers report and match to existing PDFs.

Creates ranked_papers.json for each topic with:
- Paper metadata (title, URL, citations, mentions)
- Path to existing PDF if found
- Score components for ranking

Usage:
    # Process single topic
    python scripts/presentations/gather_topic_papers.py --topic 01

    # Process all topics
    python scripts/presentations/gather_topic_papers.py --all

Report format (conferences/neurips2025/agent-work/claude/paper-influence/reports/):
    ## 1. Efficient Memory Management for LLM Serving (vLLM)
    - **Authors:** Woosuk Kwon, ...
    - **Venue:** SOSP 2023
    - **URL:** https://arxiv.org/abs/2309.06180
    - **GitHub:** https://github.com/vllm-project/vllm
    **Analytical Scores:**
    - **Citations:** 2,500+ (...)
    - **Mentions (4):** ...

PDF naming convention (conferences/neurips2025/all-notable-papers/):
    {topic}__rank{NN}__{slug}__{hash}.pdf
    Example: 01_tools_and_open_source__rank03__efficient-memory-vllm__65ff338f.pdf
"""

import argparse
import json
import re
from pathlib import Path


# Topic name mapping (number -> full name for report files)
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

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "claude-autonomy" / "paper-influence" / "reports"
PAPERS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "all-notable-papers"
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"


def parse_influential_papers(report_path: Path) -> list[dict]:
    """Parse markdown report into structured paper list."""
    content = report_path.read_text()
    papers = []

    # Split content by --- separators to get individual paper sections
    sections = re.split(r'\n---\n', content)

    for section in sections:
        # Find paper header: ## 1. Paper Title or ## 2. Paper Title
        header_match = re.search(r'## (\d+)\. (.+?)(?:\n|$)', section)
        if not header_match:
            continue

        rank = int(header_match.group(1))
        title = header_match.group(2).strip()

        # Extract URL (arXiv or other)
        url_match = re.search(r'\*\*URL:\*\*\s*(https?://[^\s\n]+)', section)
        arxiv_url = url_match.group(1).strip() if url_match else None

        # Extract GitHub URL
        github_match = re.search(r'\*\*GitHub:\*\*\s*(https?://[^\s\n]+)', section)
        github_url = github_match.group(1).strip() if github_match else None

        # Extract authors
        authors_match = re.search(r'\*\*Authors:\*\*\s*(.+?)(?:\n|$)', section)
        authors = authors_match.group(1).strip() if authors_match else None

        # Extract venue
        venue_match = re.search(r'\*\*Venue:\*\*\s*(.+?)(?:\n|$)', section)
        venue = venue_match.group(1).strip() if venue_match else None

        # Extract citations count
        citations_match = re.search(r'\*\*Citations:\*\*\s*([\d,]+)', section)
        citations = int(citations_match.group(1).replace(',', '')) if citations_match else 0

        # Extract mentions count
        mentions_match = re.search(r'\*\*Mentions \((\d+)\)', section)
        mentions = int(mentions_match.group(1)) if mentions_match else 0

        # Extract published date/year
        published_match = re.search(r'\*\*Published:\*\*\s*(.+?)(?:\n|$)', section)
        published = published_match.group(1).strip() if published_match else None

        # Extract recency score if present
        recency_match = re.search(r'\*\*Recency Score:\*\*\s*([\d.]+)', section)
        recency_score = float(recency_match.group(1)) if recency_match else 0.5

        papers.append({
            "rank": rank,
            "title": title,
            "authors": authors,
            "venue": venue,
            "arxiv_url": arxiv_url,
            "github_url": github_url,
            "citations": citations,
            "mentions": mentions,
            "published": published,
            "recency_score": recency_score,
        })

    return papers


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching."""
    # Lowercase and remove special characters
    normalized = re.sub(r'[^a-z0-9\s]', '', title.lower())
    # Remove common words
    stopwords = {'the', 'a', 'an', 'for', 'of', 'and', 'in', 'on', 'with', 'to', 'is', 'are'}
    words = [w for w in normalized.split() if w not in stopwords]
    return ' '.join(words)


def title_similarity(title1: str, title2: str) -> float:
    """Calculate simple word overlap similarity between two titles."""
    words1 = set(normalize_title(title1).split())
    words2 = set(normalize_title(title2).split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def find_pdf_for_paper(paper: dict, topic_num: str, pdf_cache: list[Path] | None = None) -> Path | None:
    """
    Find matching PDF using title similarity.

    PDF naming pattern: {topic}__rank{NN}__{slug}__{hash}.pdf
    The slug in the filename contains a simplified version of the paper title.
    """
    if not PAPERS_DIR.exists():
        return None

    # Get all PDFs for this topic
    if pdf_cache is None:
        pdf_cache = list(PAPERS_DIR.glob(f"{topic_num}_*.pdf"))

    if not pdf_cache:
        return None

    paper_title = paper['title']
    best_match = None
    best_score = 0.0

    for pdf_path in pdf_cache:
        # Extract slug from filename: {topic}__rank{NN}__{slug}__{hash}.pdf
        filename = pdf_path.stem  # without .pdf
        parts = filename.split('__')
        if len(parts) >= 3:
            # The slug is in parts[2], convert dashes to spaces
            slug = parts[2].replace('-', ' ')
            score = title_similarity(paper_title, slug)

            if score > best_score:
                best_score = score
                best_match = pdf_path

    # Only return match if similarity is above threshold
    if best_score >= 0.3:  # At least 30% word overlap
        return best_match

    return None


def calculate_scores(paper: dict, max_citations: int, max_mentions: int) -> dict:
    """
    Calculate weighted scores for paper ranking.

    Weights (from Presentation_building_plan.md Part 7.1):
    - Conference Mention Frequency: 25%
    - Keynote/Spotlight Status: 30% (estimated from venue/mentions)
    - Citation Count: 20%
    - Recency: 5%
    - Topic Relevance: 20% (estimated as 1.0 for papers in this list)
    """
    # Normalize mentions (0-1)
    mention_score = paper['mentions'] / max_mentions if max_mentions > 0 else 0

    # Estimate keynote score based on mentions (high mentions = likely featured)
    keynote_score = min(1.0, paper['mentions'] / 5) if paper['mentions'] > 0 else 0

    # Normalize citations (0-1)
    citation_score = paper['citations'] / max_citations if max_citations > 0 else 0

    # Recency score (from report or default 0.5)
    recency_score = paper.get('recency_score', 0.5)

    # Relevance (assume 1.0 since they're in the influential papers list)
    relevance_score = 1.0

    # Calculate weighted total
    total = (
        0.25 * mention_score +
        0.30 * keynote_score +
        0.20 * citation_score +
        0.05 * recency_score +
        0.20 * relevance_score
    )

    return {
        "mention": round(mention_score, 3),
        "keynote": round(keynote_score, 3),
        "citation": round(citation_score, 3),
        "recency": round(recency_score, 3),
        "relevance": round(relevance_score, 3),
        "total": round(total, 3),
    }


def process_topic(topic_num: str) -> dict:
    """Process a single topic and create ranked_papers.json."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic number: {topic_num}")

    # Find report file
    report_path = REPORTS_DIR / f"{topic_name}_influential_papers.md"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    print(f"\nProcessing: {topic_name}")
    print(f"  Report: {report_path.name}")

    # Parse papers from report
    papers = parse_influential_papers(report_path)
    print(f"  Found {len(papers)} papers in report")

    # Calculate max values for normalization
    max_citations = max((p['citations'] for p in papers), default=1)
    max_mentions = max((p['mentions'] for p in papers), default=1)

    # Cache PDFs for this topic (avoid re-globbing for each paper)
    pdf_cache = list(PAPERS_DIR.glob(f"{topic_num}_*.pdf")) if PAPERS_DIR.exists() else []
    print(f"  Available PDFs: {len(pdf_cache)}")

    # Match PDFs and calculate scores
    matched_count = 0
    for paper in papers:
        # Find PDF by title similarity
        pdf_path = find_pdf_for_paper(paper, topic_num, pdf_cache)
        if pdf_path:
            paper['pdf_path'] = str(pdf_path)
            paper['pdf_exists'] = True
            matched_count += 1
        else:
            paper['pdf_path'] = None
            paper['pdf_exists'] = False

        # Calculate scores
        paper['score'] = calculate_scores(paper, max_citations, max_mentions)

    print(f"  Matched {matched_count}/{len(papers)} PDFs")

    # Create output
    output = {
        "topic": topic_name,
        "topic_num": topic_num,
        "source_report": str(report_path.relative_to(PROJECT_ROOT)),
        "papers_count": len(papers),
        "pdfs_matched": matched_count,
        "papers": papers,
    }

    # Save to topic directory
    output_dir = PRESENTATIONS_DIR / topic_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ranked_papers.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Gather papers for presentation topics")
    parser.add_argument("--topic", type=str, help="Topic number (01-20)")
    parser.add_argument("--all", action="store_true", help="Process all topics")
    args = parser.parse_args()

    if args.all:
        # Process all topics
        print("Processing all topics...")
        results = {}
        for topic_num in TOPIC_NAMES.keys():
            try:
                results[topic_num] = process_topic(topic_num)
            except Exception as e:
                print(f"  Error: {e}")
                results[topic_num] = {"error": str(e)}

        # Summary
        print("\n" + "=" * 60)
        print("Summary:")
        total_papers = sum(r.get('papers_count', 0) for r in results.values())
        total_matched = sum(r.get('pdfs_matched', 0) for r in results.values())
        print(f"  Total papers: {total_papers}")
        print(f"  PDFs matched: {total_matched}")
        print(f"  Match rate: {total_matched/total_papers*100:.1f}%")

    elif args.topic:
        # Normalize topic number
        topic_num = args.topic.zfill(2)
        process_topic(topic_num)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
