#!/usr/bin/env python3
"""
Setup GPT Researcher for NeurIPS Conference Pipeline

This script indexes all presentations from the conference data directory
and prepares them for research queries using GPT Researcher.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conference_pipeline.model_governance.registry import default_model_for_provider


def load_env_from_environment():
    """Return the API keys set in the environment.

    Keys are never read from a shell profile: source the repository's .env first
    (set -a; source .env; set +a).
    """
    return {key: value for key, value in os.environ.items() if key.endswith("_API_KEY")}


def setup_environment():
    """Setup environment variables for GPT Researcher"""
    env_vars = load_env_from_environment()

    # Set required environment variables
    if 'OPENAI_API_KEY' in env_vars:
        os.environ['OPENAI_API_KEY'] = env_vars['OPENAI_API_KEY']
        print("✓ Loaded OPENAI_API_KEY")
    else:
        print("✗ OPENAI_API_KEY not set in the environment (source the repository .env)")

    if 'TAVILY_API_KEY' in env_vars:
        os.environ['TAVILY_API_KEY'] = env_vars['TAVILY_API_KEY']
        print("✓ Loaded TAVILY_API_KEY")
    else:
        print("✗ TAVILY_API_KEY not set in the environment (source the repository .env)")


def collect_presentation_data(presentations_dir: Path) -> List[Dict[str, Any]]:
    """
    Collect all presentation data from the presentations directory.

    Returns a list of presentation documents with their content.
    """
    presentations = []

    if not presentations_dir.exists():
        print(f"Presentations directory not found: {presentations_dir}")
        return presentations

    for pres_dir in sorted(presentations_dir.iterdir()):
        if not pres_dir.is_dir():
            continue

        print(f"Processing: {pres_dir.name}")

        pres_data = {
            'id': pres_dir.name,
            'path': str(pres_dir),
            'meta': {},
            'transcript': '',
            'slides': [],
            'references': []
        }

        # Load meta.json
        meta_file = pres_dir / 'meta.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                pres_data['meta'] = json.load(f)

        # Load transcript
        transcript_file = pres_dir / 'transcripts' / 'transcript.txt'
        if transcript_file.exists():
            with open(transcript_file, 'r') as f:
                pres_data['transcript'] = f.read()

        # Load OCR data from slides
        ocr_dir = pres_dir / 'ocr'
        if ocr_dir.exists():
            for ocr_file in sorted(ocr_dir.glob('slide_*.json')):
                try:
                    with open(ocr_file, 'r') as f:
                        slide_data = json.load(f)
                        pres_data['slides'].append(slide_data)

                        # Collect references
                        if 'references' in slide_data and slide_data['references']:
                            pres_data['references'].extend(slide_data['references'])
                except json.JSONDecodeError:
                    print(f"  Warning: Could not parse {ocr_file}")
                    continue

        presentations.append(pres_data)

    print(f"\n✓ Collected {len(presentations)} presentations")
    return presentations


def create_document_index(presentations: List[Dict[str, Any]], output_dir: Path):
    """
    Create a document index for GPT Researcher.

    This creates structured text documents from presentations that can be
    used as a local knowledge base.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a documents directory
    docs_dir = output_dir / 'documents'
    docs_dir.mkdir(exist_ok=True)

    # Create master index
    index = {
        'presentations': [],
        'total_count': len(presentations)
    }

    for pres in presentations:
        pres_id = pres['id']
        meta = pres['meta']

        # Create a comprehensive document for this presentation
        doc_path = docs_dir / f"{pres_id}.txt"

        with open(doc_path, 'w') as f:
            # Write header
            f.write(f"# {meta.get('title', pres_id)}\n\n")

            # Write metadata
            f.write("## Metadata\n\n")
            f.write(f"- ID: {meta.get('id', 'N/A')}\n")
            f.write(f"- Track: {meta.get('track', 'N/A')}\n")
            f.write(f"- NeurIPS URL: {meta.get('neurips_url', 'N/A')}\n")
            f.write(f"- SlidesLive URL: {meta.get('slideslive_url', 'N/A')}\n")
            if meta.get('authors'):
                f.write(f"- Authors: {', '.join(meta['authors'])}\n")
            f.write("\n")

            # Write slide content
            f.write("## Slide Content\n\n")
            for slide in pres['slides']:
                slide_text = slide.get('slide_text', {})

                if slide_text.get('title'):
                    f.write(f"### Slide {slide.get('slide_index', '?')}: {slide_text['title']}\n\n")
                else:
                    f.write(f"### Slide {slide.get('slide_index', '?')}\n\n")

                # Write equations
                if slide_text.get('equations'):
                    f.write("**Equations:**\n")
                    for eq in slide_text['equations']:
                        f.write(f"- {eq}\n")
                    f.write("\n")

                # Write bullet points
                if slide_text.get('bullet_points'):
                    for bullet in slide_text['bullet_points']:
                        f.write(f"- {bullet}\n")
                    f.write("\n")

                # Write other text
                if slide_text.get('other_text'):
                    f.write(f"{slide_text['other_text']}\n\n")

                # Write key concepts
                if slide.get('key_concepts'):
                    f.write("**Key Concepts:**\n")
                    for concept in slide['key_concepts']:
                        f.write(f"- {concept.get('concept', '')}: {concept.get('explanation', '')}\n")
                    f.write("\n")

            # Write transcript
            if pres['transcript']:
                f.write("## Full Transcript\n\n")
                f.write(pres['transcript'])
                f.write("\n\n")

            # Write references
            if pres['references']:
                f.write("## References\n\n")
                unique_refs = list({ref.get('title', ref.get('text', ''))
                                   for ref in pres['references'] if ref})
                for ref in unique_refs:
                    if ref:
                        f.write(f"- {ref}\n")
                f.write("\n")

        # Add to index
        index['presentations'].append({
            'id': pres_id,
            'title': meta.get('title', pres_id),
            'track': meta.get('track', ''),
            'document_path': str(doc_path),
            'has_transcript': bool(pres['transcript']),
            'num_slides': len(pres['slides']),
            'num_references': len(pres['references'])
        })

    # Write index file
    index_path = output_dir / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"✓ Created document index at {output_dir}")
    print(f"  - {len(presentations)} documents")
    print(f"  - Index: {index_path}")
    print(f"  - Documents: {docs_dir}")


def create_gpt_researcher_config(output_dir: Path):
    """
    Create a configuration file for GPT Researcher to use local documents.
    """
    config = {
        "retriever": "local",
        "local_docs_path": str(output_dir / "documents"),
        "embedding_provider": "openai",
        "llm_provider": "openai",
        "fast_llm_model": default_model_for_provider("research.gpt_researcher_fast", "openai"),
        "smart_llm_model": default_model_for_provider("research.gpt_researcher_smart", "openai"),
        "temperature": 0.2,
        "max_search_results": 10,
        "browse_chunk_max_length": 8000,
        "summary_token_limit": 1000,
    }

    config_path = output_dir / 'gpt_researcher_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created GPT Researcher config at {config_path}")
    return config_path


def main():
    """Main execution function"""
    print("=" * 60)
    print("GPT Researcher Setup for Conference Pipeline")
    print("=" * 60)
    print()

    # Setup environment
    print("Step 1: Loading environment variables...")
    setup_environment()
    print()

    # Define paths
    project_root = Path(__file__).parent.parent
    presentations_dir = project_root / 'conferences' / 'neurips2025' / 'data' / 'presentations'
    output_dir = project_root / 'conferences' / 'neurips2025' / 'gpt_researcher'

    # Collect presentations
    print("Step 2: Collecting presentation data...")
    presentations = collect_presentation_data(presentations_dir)
    print()

    if not presentations:
        print("No presentations found. Exiting.")
        return

    # Create document index
    print("Step 3: Creating document index...")
    create_document_index(presentations, output_dir)
    print()

    # Create config
    print("Step 4: Creating GPT Researcher configuration...")
    config_path = create_gpt_researcher_config(output_dir)
    print()

    # Print summary
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print(f"Indexed presentations: {len(presentations)}")
    print(f"Output directory: {output_dir}")
    print()
    print("To use GPT Researcher with this index:")
    print()
    print("  from gpt_researcher import GPTResearcher")
    print("  import asyncio")
    print()
    print("  async def research():")
    print("      researcher = GPTResearcher(")
    print("          query='Your research question',")
    print("          report_type='research_report',")
    print(f"          config_path='{config_path}'")
    print("      )")
    print("      report = await researcher.conduct_research()")
    print("      return report")
    print()
    print("  asyncio.run(research())")
    print()


if __name__ == '__main__':
    main()
