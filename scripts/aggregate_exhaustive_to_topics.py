#!/usr/bin/env python3
"""
Aggregate exhaustive extractions by topic and generate Gemini summaries.

For each topic:
1. Read the JSONL file from exhaustive_extractions/presentations/by_topic/
2. Aggregate verbatim quotes and summaries
3. Send to Gemini 3 Pro Preview for synthesis
4. Append the result to the corresponding analysis document
"""

import json
import sys
import warnings
from pathlib import Path

import google.generativeai as genai

from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    legacy_gemini_generate_content,
)

SUMMARY_MODEL = default_model_for_provider("synthesis.exhaustive_topic_summary", "gemini")

warnings.filterwarnings('ignore')

# Topic mapping
TOPICS = {
    "01": ("Tools and Open Source", "01_tools_and_open_source.md"),
    "02": ("GenAI Architectures", "02_genai_architectures.md"),
    "03": ("Data Preparation", "03_data_preparation.md"),
    "04": ("Data Attribution", "04_data_attribution.md"),
    "05": ("Data Markets", "05_data_markets.md"),
    "06": ("Visual Models", "06_visual_models.md"),
    "07": ("Tabular Models", "07_tabular_models.md"),
    "08": ("Language Models", "08_language_models.md"),
    "09": ("GenAI in Science", "09_genai_in_science.md"),
    "10": ("Serendipitous Findings", "10_serendipitous_findings.md"),
    "11": ("Ethical Problems", "11_ethical_problems.md"),
    "12": ("Commercial Applications", "12_commercial_applications.md"),
    "13": ("Industry Insights", "13_industry_insights.md"),
    "14": ("Synthetic Data", "14_synthetic_data.md"),
    "15": ("What's Happening at NeurIPS", "15_whats_happening_at_neurips.md"),
    "16": ("NeurIPS Awards", "16_neurips_awards.md"),
    "17": ("LLM Scaling Laws & Efficiency", "17_llm_scaling_laws_efficiency.md"),
    "18": ("Agentic AI & Autonomous Systems", "18_agentic_ai_autonomous_systems.md"),
    "19": ("Notable Quotes", "19_notable_quotes.md"),
    "20": ("Small Language Models", "20_small_language_models.md"),
}

def aggregate_jsonl(jsonl_path: Path) -> str:
    """Read JSONL and aggregate into markdown content."""
    extractions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                extractions.append(json.loads(line))

    # Group by source/talk for better organization
    by_source = {}
    for ext in extractions:
        # Use talk info if available, otherwise source
        if ext.get('talk_title'):
            key = f"{ext.get('talk_title', 'Unknown')} - {ext.get('talk_speaker', 'Unknown Speaker')}"
        else:
            key = ext.get('source_title', 'Unknown Source')

        if key not in by_source:
            by_source[key] = []
        by_source[key].append(ext)

    # Build markdown content
    lines = []
    for source, exts in sorted(by_source.items()):
        lines.append(f"\n## {source}\n")
        for ext in exts:
            timestamp = ext.get('timestamp_start', '')
            if ext.get('verbatim_quote'):
                lines.append(f"**[{timestamp}]** \"{ext['verbatim_quote']}\"\n")
            lines.append(f"- {ext.get('summary', '')}\n")

    return "\n".join(lines)


def generate_summary(topic_name: str, content: str, api_key: str) -> str:
    """Send content to the governed Gemini summary model and get summary."""
    genai.configure(api_key=api_key)

    prompt = f"""The following is the transcript from talks on NeurIPS from talks that talk about {topic_name}. I want you to create a summary highlighting the main interesting facts, discoveries, breakthroughs, achievements and trends in the domain. Feel free to create a document to put your thoughts. You can organize your thoughts in sections. Make sure that for any claims you reference names of people publications workshop names and presentation titles. Imagine that you are going to speak to an audience of scientists who didn't attend the conference and they are looking to learn about what happened

{content}"""

    model = genai.GenerativeModel(SUMMARY_MODEL)
    response = legacy_gemini_generate_content(model, prompt, caller=__file__)
    return response.text


def append_to_analysis(analysis_path: Path, summary: str, topic_name: str):
    """Append Gemini summary to the analysis document."""
    section = f"""

---

## Exhaustive Extraction Summary (Gemini 3 Pro)

*Auto-generated summary from exhaustive extraction pipeline*

{summary}
"""

    with open(analysis_path, 'a') as f:
        f.write(section)


def log(msg):
    """Print with flush for immediate output."""
    print(msg, flush=True)


def main():
    base_dir = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
    extractions_dir = base_dir / "exhaustive_extractions" / "presentations" / "by_topic"
    analysis_dir = base_dir / "analysis"

    api_key = "REDACTED_GCP_API_KEY"

    # Process specific topic if provided, otherwise all
    topics_to_process = sys.argv[1:] if len(sys.argv) > 1 else list(TOPICS.keys())

    for topic_id in topics_to_process:
        if topic_id not in TOPICS:
            print(f"Unknown topic: {topic_id}")
            continue

        topic_name, analysis_file = TOPICS[topic_id]
        jsonl_path = extractions_dir / f"{topic_id}.jsonl"
        analysis_path = analysis_dir / analysis_file

        if not jsonl_path.exists():
            log(f"[{topic_id}] No JSONL file found, skipping")
            continue

        if not analysis_path.exists():
            log(f"[{topic_id}] No analysis file found, skipping")
            continue

        log(f"[{topic_id}] Processing {topic_name}...")

        # Aggregate content
        log(f"  - Aggregating extractions from {jsonl_path.name}...")
        content = aggregate_jsonl(jsonl_path)
        content_size = len(content)
        log(f"  - Aggregated {content_size:,} characters")

        # Check if content is too large (Gemini has limits)
        if content_size > 2_000_000:
            log(f"  - Content too large, truncating to 2M chars")
            content = content[:2_000_000]

        # Generate summary
        log(f"  - Sending to Gemini 3 Pro Preview...")
        try:
            summary = generate_summary(topic_name, content, api_key)
            log(f"  - Received {len(summary):,} character summary")
        except Exception as e:
            log(f"  - ERROR: {e}")
            continue

        # Append to analysis
        log(f"  - Appending to {analysis_file}...")
        append_to_analysis(analysis_path, summary, topic_name)

        log(f"[{topic_id}] Done!")
        log("")


if __name__ == "__main__":
    main()
