#!/usr/bin/env python3
"""
Phase 6: Fill coverage gaps by adding references to topic documents.

This script:
1. Loads the presentation-to-topic mapping
2. Applies manual corrections for misclassified presentations
3. For each topic, adds an "Additional Presentations" section with references
4. Extracts brief content snippets from transcripts for meaningful references
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Manual corrections for misclassified presentations
TOPIC_CORRECTIONS = {
    # Biology/Science misclassified as visual
    "imageomics-discovering-biological-knowledge-from-images-using-ai": "09_genai_in_science",
    "building-an-ai-ecosystem-for-multiscale-biological-discovery": "09_genai_in_science",

    # Drug/Medical to Science
    "cure-bench-competition-on-reasoning-models-for-drug-decision-making-in": "09_genai_in_science",
    "qoq-med-building-multimodal-clinical-foundation-models-with-domain-aware-grpo": "09_genai_in_science",
    "2nd-workshop-on-multi-modal-foundation-models-and-large-language-models-for": "09_genai_in_science",

    # Physics/Cosmology to Science
    "fair-universe-handling-uncertainties-and-distribution-shifts-for-precision": "09_genai_in_science",

    # Algorithms/Architectures misclassified as serendipitous
    "differentiable-learning-of-combinatorial-algorithms-from-theory-to-practice": "02_genai_architectures",
    "recent-developments-in-geometric-machine-learning-foundations-models-and-more": "02_genai_architectures",
    "adjoint-schrödinger-bridge-sampler": "02_genai_architectures",
    "gnnxemplar-exemplars-to-explanations-natural-language-rules-for-global-gnn": "02_genai_architectures",
    "dynamical-decoupling-of-generalization-and-overfitting-in-large-two-layer": "02_genai_architectures",
    "rag4gfm-bridging-knowledge-gaps-in-graph-foundation-models-through-graph": "02_genai_architectures",

    # Efficiency/Optimization corrections
    "auto-compressing-networks": "17_llm_scaling_laws_efficiency",
    "in-search-of-adam-s-secret-sauce": "17_llm_scaling_laws_efficiency",

    # Ethics corrections
    "real-time-hyper-personalized-generative-ai-should-be-regulated-to-prevent-the": "11_ethical_problems",

    # Industry/Autonomous driving
    "toward-general-full-autonomy-open-research-challenges-in-scalable-self-driving": "13_industry_insights",

    # Agents corrections
    "ai-assistants-in-the-wild-agents-adaptation-and-memory-augmented-deployment": "18_agentic_ai_autonomous_systems",

    # Code/LLM
    "the-2025-google-code-golf-championship": "08_language_models",

    # Multimodal to Visual
    "multimodal-data-foundation-at-industry-scale": "12_commercial_applications",
    "telling-stories-at-scale-multimodal-ml-in-the-global-media-landscape": "12_commercial_applications",
}


def load_mapping(mapping_file: Path) -> List[Dict]:
    """Load presentation-to-topic mapping."""
    with open(mapping_file) as f:
        data = json.load(f)
    return data["mappings"]


def apply_corrections(mappings: List[Dict]) -> List[Dict]:
    """Apply manual corrections to mappings."""
    for m in mappings:
        if m["folder"] in TOPIC_CORRECTIONS:
            old_topic = m["topic"]
            m["topic"] = TOPIC_CORRECTIONS[m["folder"]]
            print(f"  Corrected: {m['folder'][:50]}... {old_topic} → {m['topic']}")
    return mappings


def extract_snippet(presentation_dir: Path, max_words: int = 50) -> str:
    """Extract a brief content snippet from a presentation transcript."""
    transcript_file = presentation_dir / "transcripts" / "full_transcript.txt"

    if not transcript_file.exists():
        return ""

    try:
        with open(transcript_file) as f:
            content = f.read()

        # Skip intro/welcome text, find substantive content
        lines = content.split('\n')
        substantive_lines = []

        for line in lines[5:]:  # Skip first few lines (often intro)
            line = line.strip()
            if len(line) > 50 and not any(skip in line.lower() for skip in
                ["welcome", "thank you", "good morning", "good afternoon", "hello everyone"]):
                substantive_lines.append(line)
                if len(' '.join(substantive_lines).split()) > max_words:
                    break

        if substantive_lines:
            text = ' '.join(substantive_lines)
            words = text.split()[:max_words]
            return ' '.join(words) + "..."

    except Exception as e:
        pass

    return ""


def format_duration(minutes: float) -> str:
    """Format duration in human-readable format."""
    if minutes == 0:
        return "N/A"
    elif minutes >= 60:
        return f"{minutes/60:.1f} hours"
    else:
        return f"{int(minutes)} min"


def generate_topic_section(presentations: List[Dict], presentations_dir: Path) -> str:
    """Generate the Additional Presentations section for a topic."""
    lines = []
    lines.append("\n---\n")
    lines.append("## Additional Presentations (Phase 6 Coverage)\n")
    lines.append("The following presentations were identified as relevant to this topic:\n")

    # Sort by duration (longest first)
    presentations = sorted(presentations, key=lambda x: -x["duration_minutes"])

    for p in presentations:
        folder = p["folder"]
        title = p["title"]
        duration = format_duration(p["duration_minutes"])

        lines.append(f"### `{folder}`\n")
        lines.append(f"**Title:** {title}")
        lines.append(f"**Duration:** {duration}\n")

        # Try to get a snippet
        pres_dir = presentations_dir / folder
        snippet = extract_snippet(pres_dir)
        if snippet:
            lines.append(f"> {snippet}\n")

        lines.append("")

    return "\n".join(lines)


def update_topic_document(topic_file: Path, new_section: str) -> bool:
    """Add the new section to a topic document."""
    with open(topic_file) as f:
        content = f.read()

    # Check if section already exists
    if "## Additional Presentations (Phase 6 Coverage)" in content:
        print(f"  Section already exists in {topic_file.name}, skipping")
        return False

    # Find the best insertion point (before Cross-References or at end)
    insertion_points = [
        "## Cross-References",
        "## External Resources",
        "## References",
        "---\n\n**Document Version:",
        "---\n\n**Generated:"
    ]

    for marker in insertion_points:
        if marker in content:
            content = content.replace(marker, new_section + marker)
            break
    else:
        # Append at end
        content += new_section

    with open(topic_file, 'w') as f:
        f.write(content)

    return True


def main():
    base_dir = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
    mapping_file = base_dir / "analysis" / "presentation_topic_mapping.json"
    presentations_dir = base_dir / "decks"
    analysis_dir = base_dir / "analysis"

    print("Loading presentation-to-topic mapping...")
    mappings = load_mapping(mapping_file)
    print(f"  Loaded {len(mappings)} mappings")

    print("\nApplying manual corrections...")
    mappings = apply_corrections(mappings)

    # Group by topic
    topic_groups = defaultdict(list)
    for m in mappings:
        topic_groups[m["topic"]].append(m)

    print(f"\nProcessing {len(topic_groups)} topics...")

    updated_topics = []
    for topic, presentations in sorted(topic_groups.items()):
        topic_file = analysis_dir / f"{topic}.md"

        if not topic_file.exists():
            print(f"  ⚠️  {topic}.md not found, skipping")
            continue

        print(f"\n{topic}: {len(presentations)} presentations")

        # Generate the new section
        new_section = generate_topic_section(presentations, presentations_dir)

        # Update the document
        if update_topic_document(topic_file, new_section):
            updated_topics.append(topic)
            print(f"  ✅ Updated {topic}.md")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total presentations mapped: {len(mappings)}")
    print(f"Topics updated: {len(updated_topics)}")
    print(f"Topics: {', '.join(updated_topics)}")

    # Save corrected mapping
    corrected_mapping = {
        "total_presentations": len(mappings),
        "topic_counts": {t: len(p) for t, p in topic_groups.items()},
        "mappings": mappings
    }

    corrected_file = base_dir / "analysis" / "presentation_topic_mapping_corrected.json"
    with open(corrected_file, 'w') as f:
        json.dump(corrected_mapping, f, indent=2)
    print(f"\nCorrected mapping saved to: {corrected_file}")


if __name__ == "__main__":
    main()
