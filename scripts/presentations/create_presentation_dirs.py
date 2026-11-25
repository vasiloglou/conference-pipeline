#!/usr/bin/env python3
"""
Create presentation directory structure for all 20 NeurIPS 2025 topics.

Each topic gets:
  - assets/figures/     - For extracted paper figures
  - assets/talk_slides/ - For extracted presentation slides

Usage:
    python scripts/presentations/create_presentation_dirs.py
"""

from pathlib import Path

TOPICS = [
    "01_tools_and_open_source",
    "02_genai_architectures",
    "03_data_preparation",
    "04_data_attribution",
    "05_data_markets",
    "06_visual_models",
    "07_tabular_models",
    "08_language_models",
    "09_genai_in_science",
    "10_serendipitous_findings",
    "11_ethical_problems",
    "12_commercial_applications",
    "13_industry_insights",
    "14_synthetic_data",
    "15_whats_happening_at_neurips",
    "16_neurips_awards",
    "17_llm_scaling_laws_efficiency",
    "18_agentic_ai_autonomous_systems",
    "19_notable_quotes",
    "20_small_language_models",
]

# Base directory relative to project root
BASE_DIR = Path(__file__).parent.parent.parent / "conferences" / "neurips2025" / "presentations"


def main():
    """Create directory structure for all topics."""
    print(f"Creating presentation directories in: {BASE_DIR}")
    print("-" * 60)

    created_count = 0
    for topic in TOPICS:
        topic_dir = BASE_DIR / topic
        figures_dir = topic_dir / "assets" / "figures"
        talk_slides_dir = topic_dir / "assets" / "talk_slides"

        # Create directories
        figures_dir.mkdir(parents=True, exist_ok=True)
        talk_slides_dir.mkdir(parents=True, exist_ok=True)

        # Check if newly created or already existed
        if not (topic_dir / "ranked_papers.json").exists():
            print(f"  Created: {topic}")
            created_count += 1
        else:
            print(f"  Exists:  {topic}")

    print("-" * 60)
    print(f"Total: {len(TOPICS)} topics ({created_count} new)")

    # Also create the master presentation directory for later
    master_dir = BASE_DIR / "master" / "assets" / "figures"
    master_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAlso created: master/assets/figures/")


if __name__ == "__main__":
    main()
