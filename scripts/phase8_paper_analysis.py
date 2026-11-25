#!/usr/bin/env python3
"""
Phase 8: Paper-Level Analysis for NeurIPS 2025

Phase 8a: Score papers by importance
Phase 8b: Classify papers into topics

Usage:
    .venv/bin/python scripts/phase8_paper_analysis.py --score    # Run scoring
    .venv/bin/python scripts/phase8_paper_analysis.py --classify # Run classification
    .venv/bin/python scripts/phase8_paper_analysis.py --all      # Run both
"""

import json
import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional
import chromadb

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "conferences/neurips2025/data"
ANALYSIS_DIR = BASE_DIR / "conferences/neurips2025/analysis"
VECTOR_STORE_DIR = BASE_DIR / "conferences/neurips2025/cache/vector-store"

# Major AI Labs for affiliation scoring
MAJOR_LABS = [
    "google", "deepmind", "anthropic", "openai", "meta", "microsoft",
    "nvidia", "apple", "amazon", "ibm", "hugging face", "huggingface",
    "alibaba", "tencent", "baidu", "bytedance", "salesforce",
    "stanford", "mit", "berkeley", "cmu", "princeton", "harvard", "oxford", "cambridge"
]

# Priority keywords for topic matching (from Phase 8 plan)
HIGH_PRIORITY_KEYWORDS = [
    "scaling law", "in-context learning", "reasoning", "agent", "agentic",
    "diffusion", "transformer", "attention", "moe", "mixture of experts",
    "hallucination", "alignment", "rlhf", "dpo", "synthetic data",
    "world model", "multimodal", "vision-language", "code generation",
    "chain of thought", "cot", "reinforcement learning from human feedback",
    "direct preference optimization", "llm", "large language model",
    "foundation model", "pre-training", "fine-tuning", "prompt"
]

MEDIUM_PRIORITY_KEYWORDS = [
    "fine-tuning", "pruning", "quantization", "distillation", "retrieval",
    "long context", "memory", "continual learning", "graph neural",
    "reinforcement learning", "benchmark", "evaluation", "tokenization",
    "embedding", "representation learning", "self-supervised", "contrastive",
    "generative", "autoregressive", "variational", "neural network"
]

# Topic definitions for classification
TOPICS = {
    "01": {"name": "Tools and Open Source", "keywords": ["tool", "framework", "library", "open source", "infrastructure", "benchmark", "dataset"]},
    "02": {"name": "GenAI Architectures", "keywords": ["architecture", "transformer", "attention", "diffusion", "tokenization", "embedding", "neural network design"]},
    "03": {"name": "Data Preparation", "keywords": ["data cleaning", "data quality", "dataset", "data curation", "preprocessing", "distillation"]},
    "04": {"name": "Data Attribution", "keywords": ["attribution", "provenance", "data contribution", "influence function", "training data"]},
    "05": {"name": "Data Markets", "keywords": ["data valuation", "data pricing", "data marketplace", "data economy", "compensation"]},
    "06": {"name": "Visual Models", "keywords": ["vision", "image", "video", "visual", "computer vision", "image generation", "3d", "multimodal vision"]},
    "07": {"name": "Tabular Models", "keywords": ["tabular", "graph neural", "relational", "structured data", "gnn", "text-to-sql", "database"]},
    "08": {"name": "Language Models", "keywords": ["language model", "llm", "nlp", "text generation", "reasoning", "in-context learning", "prompt"]},
    "09": {"name": "GenAI in Science", "keywords": ["science", "biology", "chemistry", "physics", "drug discovery", "protein", "molecule", "scientific"]},
    "10": {"name": "Serendipitous Findings", "keywords": ["unexpected", "surprising", "novel application", "creative"]},
    "11": {"name": "Ethical Problems", "keywords": ["ethics", "fairness", "bias", "discrimination", "privacy", "safety", "harm", "alignment"]},
    "12": {"name": "Commercial Applications", "keywords": ["commercial", "industry", "application", "deployment", "production", "business"]},
    "13": {"name": "Industry Insights", "keywords": ["industry", "enterprise", "real-world", "practical", "deployment"]},
    "14": {"name": "Synthetic Data", "keywords": ["synthetic data", "data generation", "simulation", "augmentation"]},
    "16": {"name": "NeurIPS Awards", "keywords": ["best paper", "award", "outstanding"]},
    "17": {"name": "LLM Scaling & Efficiency", "keywords": ["scaling", "efficiency", "pruning", "quantization", "sparsity", "compression", "inference"]},
    "18": {"name": "Agentic AI", "keywords": ["agent", "autonomous", "planning", "tool use", "multi-agent", "embodied", "robotics"]},
    "20": {"name": "Small Language Models", "keywords": ["small model", "slm", "efficient", "on-device", "edge", "mobile", "lightweight"]},
}


def load_papers() -> list:
    """Load papers from papers.json"""
    papers_file = DATA_DIR / "papers.json"
    with open(papers_file) as f:
        data = json.load(f)
    return data["papers"]


def load_awards() -> set:
    """Load award-winning paper titles from 16_neurips_awards.md"""
    awards_file = ANALYSIS_DIR / "16_neurips_awards.md"
    award_titles = set()

    if awards_file.exists():
        content = awards_file.read_text().lower()
        # Extract paper titles mentioned in the awards document
        # Looking for patterns like "**Paper:**" or paper titles in quotes
        lines = content.split('\n')
        for line in lines:
            if 'best paper' in line or 'outstanding' in line or 'runner-up' in line:
                # Extract any quoted titles
                matches = re.findall(r'"([^"]+)"', line)
                award_titles.update(m.lower() for m in matches)

    return award_titles


def check_chromadb_mentions(paper_title: str) -> int:
    """Check if paper is mentioned in ChromaDB transcripts"""
    try:
        chroma_path = VECTOR_STORE_DIR / "chroma_db"
        if not chroma_path.exists():
            return 0

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("neurips_presentations")

        # Search for paper title mentions
        results = collection.query(
            query_texts=[paper_title],
            n_results=5
        )

        # Count relevant matches (distance < 0.5 indicates good match)
        if results and results['distances']:
            mentions = sum(1 for d in results['distances'][0] if d < 0.5)
            return mentions
        return 0
    except Exception as e:
        print(f"ChromaDB error: {e}")
        return 0


def score_paper(paper: dict, award_titles: set, check_chromadb: bool = False) -> dict:
    """
    Score a paper based on importance signals.

    Weights:
    - Award Status: 10x
    - Oral/Spotlight: 5x (oral) / 3x (spotlight)
    - Major Lab Affiliation: 3x
    - Keyword Match: 2x (high) / 1x (medium)
    - Referenced in Transcripts: 3x (if check_chromadb=True)
    """
    score = 1.0
    signals = []

    title_lower = (paper.get("title") or "").lower()
    abstract_lower = (paper.get("abstract") or "").lower()
    venue = (paper.get("venue") or "").lower()
    authors = paper.get("authors") or []
    keywords = [k.lower() for k in (paper.get("keywords") or [])]
    text = f"{title_lower} {abstract_lower} {' '.join(keywords)}"

    # 1. Award Status (10x)
    if title_lower in award_titles or any(t in title_lower for t in award_titles if len(t) > 10):
        score *= 10
        signals.append("award_winner")

    # 2. Oral/Spotlight presentation (5x / 3x)
    if "oral" in venue:
        score *= 5
        signals.append("oral_presentation")
    elif "spotlight" in venue:
        score *= 3
        signals.append("spotlight_presentation")

    # 3. Major Lab Affiliation (3x)
    # Check authors and any affiliation info in the paper
    author_text = " ".join(authors).lower()
    for lab in MAJOR_LABS:
        if lab in author_text or lab in text:
            score *= 3
            signals.append(f"major_lab:{lab}")
            break

    # 4. High Priority Keyword Match (2x per match, max 3x total)
    high_matches = sum(1 for kw in HIGH_PRIORITY_KEYWORDS if kw in text)
    if high_matches > 0:
        multiplier = min(1 + high_matches * 0.5, 3.0)
        score *= multiplier
        signals.append(f"high_keywords:{high_matches}")

    # 5. Medium Priority Keyword Match (1.5x per match, max 2x total)
    medium_matches = sum(1 for kw in MEDIUM_PRIORITY_KEYWORDS if kw in text)
    if medium_matches > 0:
        multiplier = min(1 + medium_matches * 0.25, 2.0)
        score *= multiplier
        signals.append(f"medium_keywords:{medium_matches}")

    # 6. ChromaDB mentions (if enabled)
    if check_chromadb:
        mentions = check_chromadb_mentions(paper.get("title", ""))
        if mentions > 0:
            score *= (1 + mentions * 0.5)
            signals.append(f"transcript_mentions:{mentions}")

    return {
        "id": paper.get("id"),
        "title": paper.get("title"),
        "score": round(score, 2),
        "signals": signals,
        "venue": paper.get("venue"),
        "authors": authors[:3],  # First 3 authors
        "abstract": (paper.get("abstract") or "")[:500],  # First 500 chars
        "openreview_url": paper.get("openreview_url"),
        "primary_area": paper.get("primary_area")
    }


def classify_paper(paper: dict) -> list:
    """
    Classify a paper into one or more topics.
    Returns list of (topic_id, confidence) tuples.
    """
    title_lower = (paper.get("title") or "").lower()
    abstract_lower = (paper.get("abstract") or "").lower()
    keywords = [k.lower() for k in (paper.get("keywords") or [])]
    text = f"{title_lower} {abstract_lower} {' '.join(keywords)}"

    topic_scores = []

    for topic_id, topic_info in TOPICS.items():
        score = 0
        matched_keywords = []

        for kw in topic_info["keywords"]:
            if kw in text:
                # Weight by where the keyword appears
                if kw in title_lower:
                    score += 3
                    matched_keywords.append(f"{kw}(title)")
                elif kw in keywords:
                    score += 2
                    matched_keywords.append(f"{kw}(keywords)")
                else:
                    score += 1
                    matched_keywords.append(kw)

        if score > 0:
            confidence = "high" if score >= 5 else "medium" if score >= 2 else "low"
            topic_scores.append({
                "topic_id": topic_id,
                "topic_name": topic_info["name"],
                "score": score,
                "confidence": confidence,
                "matched_keywords": matched_keywords
            })

    # Sort by score descending
    topic_scores.sort(key=lambda x: x["score"], reverse=True)

    return topic_scores[:3]  # Return top 3 topic matches


def run_scoring(papers: list, check_chromadb: bool = False) -> list:
    """Run importance scoring on all papers"""
    print(f"Scoring {len(papers)} papers...")

    award_titles = load_awards()
    print(f"Loaded {len(award_titles)} award titles")

    scored_papers = []
    for i, paper in enumerate(papers):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(papers)} papers...")

        scored = score_paper(paper, award_titles, check_chromadb)
        scored_papers.append(scored)

    # Sort by score descending
    scored_papers.sort(key=lambda x: x["score"], reverse=True)

    return scored_papers


def run_classification(papers: list, top_n: int = 200) -> dict:
    """Run topic classification on top N papers"""
    print(f"Classifying top {top_n} papers into topics...")

    classifications = {}
    for i, paper in enumerate(papers[:top_n]):
        if i % 50 == 0:
            print(f"  Classified {i}/{top_n} papers...")

        topics = classify_paper(paper)
        classifications[paper["id"]] = {
            "title": paper["title"],
            "score": paper["score"],
            "topics": topics
        }

    return classifications


def generate_report(scored_papers: list, classifications: dict, output_dir: Path):
    """Generate Phase 8 report files"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save importance scores
    scores_file = output_dir / "importance_scores.json"
    with open(scores_file, 'w') as f:
        json.dump(scored_papers[:500], f, indent=2)  # Top 500
    print(f"Saved importance scores to {scores_file}")

    # 2. Save topic mappings
    mapping_file = output_dir / "paper_topic_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(classifications, f, indent=2)
    print(f"Saved topic mappings to {mapping_file}")

    # 3. Generate summary report
    report_file = output_dir / "phase8_report.md"

    # Count papers by topic
    topic_counts = defaultdict(list)
    for paper_id, data in classifications.items():
        for topic in data["topics"]:
            if topic["confidence"] in ["high", "medium"]:
                topic_counts[topic["topic_id"]].append({
                    "id": paper_id,
                    "title": data["title"],
                    "score": data["score"],
                    "confidence": topic["confidence"]
                })

    with open(report_file, 'w') as f:
        f.write("# Phase 8: Paper-Level Analysis Report\n\n")
        f.write(f"**Generated:** {Path(__file__).name}\n")
        f.write(f"**Papers Analyzed:** {len(scored_papers)}\n")
        f.write(f"**Papers Classified:** {len(classifications)}\n\n")

        f.write("---\n\n")
        f.write("## Top 50 Papers by Importance Score\n\n")
        f.write("| Rank | Score | Title | Venue | Signals |\n")
        f.write("|------|-------|-------|-------|--------|\n")

        for i, paper in enumerate(scored_papers[:50], 1):
            signals = ", ".join(paper["signals"][:3]) if paper["signals"] else "-"
            title_short = paper["title"][:60] + "..." if len(paper["title"]) > 60 else paper["title"]
            venue_short = paper["venue"][:20] if paper["venue"] else "-"
            f.write(f"| {i} | {paper['score']} | {title_short} | {venue_short} | {signals} |\n")

        f.write("\n---\n\n")
        f.write("## Papers by Topic\n\n")

        for topic_id in sorted(topic_counts.keys()):
            papers_in_topic = topic_counts[topic_id]
            topic_name = TOPICS[topic_id]["name"]
            f.write(f"### Topic {topic_id}: {topic_name} ({len(papers_in_topic)} papers)\n\n")

            # Show top 10 papers for this topic
            for paper in sorted(papers_in_topic, key=lambda x: x["score"], reverse=True)[:10]:
                f.write(f"- **{paper['title']}** (score: {paper['score']}, {paper['confidence']})\n")

            f.write("\n")

        f.write("\n---\n\n")
        f.write("## Distribution Statistics\n\n")

        # Venue distribution
        venue_counts = defaultdict(int)
        for paper in scored_papers[:200]:
            venue = paper["venue"] or "unknown"
            if "oral" in venue.lower():
                venue_counts["oral"] += 1
            elif "spotlight" in venue.lower():
                venue_counts["spotlight"] += 1
            elif "poster" in venue.lower():
                venue_counts["poster"] += 1
            else:
                venue_counts["other"] += 1

        f.write("### Top 200 Papers by Presentation Type\n\n")
        f.write("| Type | Count |\n")
        f.write("|------|-------|\n")
        for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {venue} | {count} |\n")

        f.write("\n### Papers per Topic (High/Medium Confidence)\n\n")
        f.write("| Topic | Count |\n")
        f.write("|-------|-------|\n")
        for topic_id in sorted(topic_counts.keys()):
            topic_name = TOPICS[topic_id]["name"]
            f.write(f"| {topic_id}: {topic_name} | {len(topic_counts[topic_id])} |\n")

    print(f"Saved report to {report_file}")

    return report_file


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Paper-Level Analysis")
    parser.add_argument("--score", action="store_true", help="Run importance scoring")
    parser.add_argument("--classify", action="store_true", help="Run topic classification")
    parser.add_argument("--all", action="store_true", help="Run both scoring and classification")
    parser.add_argument("--chromadb", action="store_true", help="Include ChromaDB mention checking (slower)")
    parser.add_argument("--top-n", type=int, default=200, help="Number of top papers to classify")

    args = parser.parse_args()

    if not (args.score or args.classify or args.all):
        args.all = True  # Default to running everything

    # Load papers
    papers = load_papers()
    print(f"Loaded {len(papers)} papers from papers.json")

    output_dir = ANALYSIS_DIR / "phase8"

    scored_papers = []
    classifications = {}

    if args.score or args.all:
        scored_papers = run_scoring(papers, check_chromadb=args.chromadb)
        print(f"\nTop 10 papers by importance score:")
        for i, paper in enumerate(scored_papers[:10], 1):
            print(f"  {i}. [{paper['score']:.1f}] {paper['title'][:70]}...")

    if args.classify or args.all:
        if not scored_papers:
            # If scoring wasn't run, use raw papers
            scored_papers = [{"id": p["id"], "title": p["title"], "score": 1.0,
                           "signals": [], "venue": p.get("venue"), "authors": p.get("authors", []),
                           "abstract": p.get("abstract", ""), "openreview_url": p.get("openreview_url"),
                           "primary_area": p.get("primary_area")} for p in papers]

        classifications = run_classification(scored_papers, top_n=args.top_n)

    if scored_papers or classifications:
        report_file = generate_report(scored_papers, classifications, output_dir)
        print(f"\n✅ Phase 8 analysis complete!")
        print(f"   Output directory: {output_dir}")
        print(f"   Report: {report_file}")


if __name__ == "__main__":
    main()
