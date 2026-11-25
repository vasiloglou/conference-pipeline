#!/usr/bin/env python3
"""
Map unreferenced presentations to existing topic documents.

This script analyzes presentation titles and content to suggest which
existing topic document each presentation should be added to.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict

# Topic mapping based on keywords
TOPIC_KEYWORDS = {
    "01_tools_and_open_source": [
        "open source", "framework", "library", "tool", "platform", "ecosystem",
        "infrastructure", "benchmark", "dataset", "demo", "tutorial"
    ],
    "02_genai_architectures": [
        "architecture", "transformer", "attention", "neural network", "deep learning",
        "diffusion", "autoencoder", "representation", "geometry", "symmetry",
        "tensor", "merging", "mode connectivity", "polynomial", "equivariant"
    ],
    "03_data_preparation": [
        "data preparation", "data quality", "data curation", "benchmark",
        "dataset", "evaluation", "data collection", "corpus"
    ],
    "04_data_attribution": [
        "attribution", "provenance", "copyright", "ownership", "credit"
    ],
    "05_data_markets": [
        "data market", "data trading", "data economy", "data sharing"
    ],
    "06_visual_models": [
        "vision", "image", "video", "visual", "cnn", "imagenet", "texture",
        "multimodal", "perception", "egocentric", "fusion"
    ],
    "07_tabular_models": [
        "tabular", "time series", "temporal", "forecasting", "relational"
    ],
    "08_language_models": [
        "language model", "llm", "reasoning", "evaluation", "benchmark",
        "trust", "reliability", "cognitive", "intelligence"
    ],
    "09_genai_in_science": [
        "science", "biology", "physics", "chemistry", "drug", "molecule",
        "protein", "genomic", "medical", "clinical", "health", "cell",
        "polymer", "material", "discovery", "neuronal", "brain"
    ],
    "10_serendipitous_findings": [
        "surprising", "unexpected", "novel", "creative"
    ],
    "11_ethical_problems": [
        "ethics", "safety", "fairness", "bias", "privacy", "regulation",
        "biosecurity", "nightmare", "harm", "risk", "causal fairness"
    ],
    "12_commercial_applications": [
        "commercial", "industry", "business", "enterprise", "e-commerce",
        "recommendation", "operational", "production", "media", "storytelling"
    ],
    "13_industry_insights": [
        "tesla", "autonomous", "self-driving", "robotics", "real-world",
        "deployment", "expo", "company"
    ],
    "14_synthetic_data": [
        "synthetic", "generation", "simulation", "fair universe"
    ],
    "17_llm_scaling_laws_efficiency": [
        "scaling", "efficiency", "optimization", "gpu", "distributed",
        "compression", "energy", "power", "accelerat"
    ],
    "18_agentic_ai_autonomous_systems": [
        "agent", "autonomous", "embodied", "planning", "decision",
        "environment", "imitation", "rl", "reinforcement", "control"
    ],
    "20_small_language_models": [
        "small language", "slm", "compact", "efficient model", "distillation"
    ]
}


@dataclass
class PresentationMapping:
    folder_name: str
    title: str
    duration_minutes: float
    suggested_topic: str
    confidence: float  # 0-1
    keywords_matched: List[str]


def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    return title.lower().replace('-', ' ')


def map_presentation_to_topic(folder_name: str, title: str) -> Tuple[str, float, List[str]]:
    """Map a presentation to the most appropriate topic."""
    normalized = normalize_title(title if title else folder_name)

    best_topic = "10_serendipitous_findings"  # Default
    best_score = 0
    best_keywords = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        matched = []
        for kw in keywords:
            if kw.lower() in normalized:
                score += 1
                matched.append(kw)

        if score > best_score:
            best_score = score
            best_topic = topic
            best_keywords = matched

    confidence = min(1.0, best_score / 3) if best_score > 0 else 0.1
    return best_topic, confidence, best_keywords


def load_unreferenced_presentations(coverage_report: Path) -> List[Tuple[str, float]]:
    """Parse coverage report to get unreferenced presentations."""
    unreferenced = []

    with open(coverage_report) as f:
        content = f.read()

    # Parse unreferenced table
    in_unreferenced = False
    for line in content.split('\n'):
        if '## Unreferenced Presentations' in line:
            in_unreferenced = True
            continue
        if in_unreferenced and line.startswith('---'):
            break
        if in_unreferenced and '`' in line:
            match = re.search(r'`([^`]+)`.*?(\d+(?:\.\d+)?)\s*(hr|min)', line)
            if match:
                folder = match.group(1)
                duration = float(match.group(2))
                if match.group(3) == 'hr':
                    duration *= 60
                unreferenced.append((folder, duration))

    return unreferenced


def load_presentation_metadata(presentations_dir: Path) -> Dict[str, str]:
    """Load presentation titles from meta.json files."""
    titles = {}

    for pres_dir in presentations_dir.iterdir():
        if not pres_dir.is_dir():
            continue

        meta_file = pres_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    titles[pres_dir.name] = meta.get("title", pres_dir.name)
            except:
                titles[pres_dir.name] = pres_dir.name.replace('-', ' ').title()
        else:
            titles[pres_dir.name] = pres_dir.name.replace('-', ' ').title()

    return titles


def main():
    base_dir = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
    coverage_report = base_dir / "analysis" / "coverage_report.md"
    presentations_dir = base_dir / "decks"

    print("Loading unreferenced presentations...")
    unreferenced = load_unreferenced_presentations(coverage_report)
    print(f"  Found {len(unreferenced)} unreferenced presentations")

    print("\nLoading presentation metadata...")
    titles = load_presentation_metadata(presentations_dir)

    print("\nMapping presentations to topics...")
    mappings = []
    topic_groups = defaultdict(list)

    for folder, duration in unreferenced:
        title = titles.get(folder, folder.replace('-', ' ').title())
        topic, confidence, keywords = map_presentation_to_topic(folder, title)

        mapping = PresentationMapping(
            folder_name=folder,
            title=title,
            duration_minutes=duration,
            suggested_topic=topic,
            confidence=confidence,
            keywords_matched=keywords
        )
        mappings.append(mapping)
        topic_groups[topic].append(mapping)

    # Print summary
    print("\n" + "="*80)
    print("MAPPING SUMMARY")
    print("="*80)

    for topic in sorted(topic_groups.keys()):
        presentations = topic_groups[topic]
        total_duration = sum(p.duration_minutes for p in presentations) / 60
        print(f"\n{topic}: {len(presentations)} presentations ({total_duration:.1f} hours)")
        for p in sorted(presentations, key=lambda x: -x.duration_minutes)[:5]:
            duration_str = f"{p.duration_minutes:.0f}min" if p.duration_minutes < 120 else f"{p.duration_minutes/60:.1f}hr"
            conf_str = f"({p.confidence:.0%})" if p.confidence < 1 else ""
            print(f"  - {p.folder_name[:60]}... {duration_str} {conf_str}")
            if p.keywords_matched:
                print(f"    Keywords: {', '.join(p.keywords_matched)}")
        if len(presentations) > 5:
            print(f"  ... and {len(presentations)-5} more")

    # Save mapping to JSON
    output = {
        "total_presentations": len(mappings),
        "topic_counts": {t: len(p) for t, p in topic_groups.items()},
        "mappings": [
            {
                "folder": m.folder_name,
                "title": m.title,
                "duration_minutes": m.duration_minutes,
                "topic": m.suggested_topic,
                "confidence": m.confidence,
                "keywords": m.keywords_matched
            }
            for m in mappings
        ]
    }

    output_file = base_dir / "analysis" / "presentation_topic_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nMapping saved to: {output_file}")

    # Print high-priority items (long presentations with low confidence)
    print("\n" + "="*80)
    print("HIGH PRIORITY: Long presentations needing manual review")
    print("="*80)

    for m in sorted(mappings, key=lambda x: -x.duration_minutes):
        if m.duration_minutes >= 60 and m.confidence < 0.5:
            print(f"\n{m.folder_name}")
            print(f"  Duration: {m.duration_minutes/60:.1f} hours")
            print(f"  Suggested: {m.suggested_topic} (confidence: {m.confidence:.0%})")
            print(f"  Keywords: {', '.join(m.keywords_matched) or 'none'}")


if __name__ == "__main__":
    main()
