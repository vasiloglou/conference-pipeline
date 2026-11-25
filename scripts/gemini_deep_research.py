"""
Gemini Deep Research using GPT-Researcher

This script uses GPT-Researcher (https://github.com/assafelovic/gpt-researcher)
configured with Google Gemini as the LLM backend to perform multi-step deep research.

Requirements:
    pip install gpt-researcher langchain-google-genai

Environment Variables Required:
    GOOGLE_API_KEY - Your Google AI API key (from ai.google.dev)
    TAVILY_API_KEY - Your Tavily API key for web search (from tavily.com)

GPT-Researcher performs actual deep research by:
1. Creating a research plan with subtopics
2. Executing multiple web searches
3. Scraping and analyzing sources
4. Synthesizing findings into a comprehensive report with citations
"""

import os
import re
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conference_pipeline.model_governance import default_model, validate_selection

# Load environment variables from .env file if present
load_dotenv()

# Configuration
BASE_DIR = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
INPUT_FILE = BASE_DIR / "analysis" / "20_small_language_models.md"
OUTPUT_DIR = BASE_DIR / "experimental_videos" / "20_small_language_models" / "gemini-deep-research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_OUTPUT_FILE = OUTPUT_DIR / "gemini_research_result.md"
SOURCES_FILE = OUTPUT_DIR / "research_sources.md"

# Configure GPT-Researcher to use Gemini
# These can also be set in a .env file
FAST_MODEL_TASK_ID = "research.gemini_deep_research_fast"
STRATEGIC_MODEL_TASK_ID = "research.gemini_deep_research_strategic"


def _gemini_researcher_model(task_id: str) -> str:
    model_name = default_model(task_id)
    validate_selection(task_id, provider="gemini", model=model_name)
    return f"google_genai:{model_name}"


os.environ.setdefault("FAST_LLM", _gemini_researcher_model(FAST_MODEL_TASK_ID))
os.environ.setdefault("SMART_LLM", _gemini_researcher_model(FAST_MODEL_TASK_ID))
os.environ.setdefault("STRATEGIC_LLM", _gemini_researcher_model(STRATEGIC_MODEL_TASK_ID))
os.environ.setdefault("EMBEDDING", "google_genai:models/text-embedding-004")


def setup_environment():
    """Verify required API keys are available."""


    missing_keys = []
    if not os.getenv("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}\n"
            f"Set them as environment variables or in a .env file.\n"
            f"Get GOOGLE_API_KEY from: https://ai.google.dev\n"
            f"Get TAVILY_API_KEY from: https://tavily.com"
        )

async def run_deep_research(query: str, report_type: str = "research_report") -> dict:
    """
    Run GPT-Researcher deep research on a query.

    Args:
        query: The research query/topic
        report_type: Type of report - "research_report", "resource_report", or "outline_report"

    Returns:
        Dictionary with report, sources, costs, and images
    """
    from gpt_researcher import GPTResearcher

    print(f"\n{'='*60}")
    print(f"Starting Deep Research with GPT-Researcher + Gemini")
    print(f"{'='*60}")
    print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
    print(f"Report Type: {report_type}")
    print(f"LLM: {os.environ.get('SMART_LLM', 'default')}")
    print(f"{'='*60}\n")

    # Initialize researcher with configuration
    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        report_format="APA",
        tone="objective and analytical",
        max_subtopics=5,
        verbose=True
    )

    # Conduct the research (this performs multiple searches and scrapes)
    print("\n[Phase 1] Conducting research (searching, scraping, analyzing)...")
    await researcher.conduct_research()

    # Generate the report
    print("\n[Phase 2] Writing comprehensive report...")
    report = await researcher.write_report()

    # Get additional data
    sources = researcher.get_research_sources()
    costs = researcher.get_costs()
    images = researcher.get_research_images()
    context = researcher.get_research_context()

    return {
        "report": report,
        "sources": sources,
        "costs": costs,
        "images": images,
        "context": context
    }


def save_results(results: dict, report_file: Path, sources_file: Path):
    """Save research results to files."""
    # Save main report
    with open(report_file, "w") as f:
        f.write(results["report"])
    print(f"\nReport saved to: {report_file}")

    # Save sources separately for easy reference
    with open(sources_file, "w") as f:
        f.write("# Research Sources\n\n")
        f.write(f"Total sources: {len(results['sources'])}\n\n")
        for i, source in enumerate(results['sources'], 1):
            f.write(f"{i}. {source}\n")
        f.write(f"\n## Research Costs\n\n")
        f.write(f"```\n{results['costs']}\n```\n")
    print(f"Sources saved to: {sources_file}")


def build_research_query(document_path: Path) -> str:
    """Build a research query from the input document."""
    with open(document_path, "r") as f:
        document_content = f.read()

    # Create a focused research query based on the document
    query = f"""
Conduct comprehensive research on Small Language Models (SLMs) and their applications,
based on the following research context from NeurIPS 2025:

{document_content}

Focus areas for deep research:
1. Find and verify specific papers mentioned - locate actual NeurIPS 2025 and arXiv papers
2. Identify key techniques: speculative decoding, knowledge distillation, quantization, pruning
3. Find recent (2024-2025) breakthrough papers on SLMs that may be missing
4. Research real-world deployment case studies and benchmarks
5. Investigate the efficiency vs capability tradeoffs in current SLM architectures

Provide verified citations with URLs for all papers and sources found.
"""
    return query


async def main():
    """Main entry point for deep research."""
    print("=" * 60)
    print("Gemini Deep Research (via GPT-Researcher)")
    print("=" * 60)

    # 1. Setup environment
    try:
        setup_environment()
        print("✓ Environment configured")
        print(f"  - FAST_LLM: {os.environ.get('FAST_LLM')}")
        print(f"  - SMART_LLM: {os.environ.get('SMART_LLM')}")
        print(f"  - STRATEGIC_LLM: {os.environ.get('STRATEGIC_LLM')}")
    except ValueError as e:
        print(f"✗ Environment setup failed:\n{e}")
        return

    # 2. Check input file
    if not INPUT_FILE.exists():
        print(f"✗ Input file not found: {INPUT_FILE}")
        return
    print(f"✓ Input file: {INPUT_FILE}")

    # 3. Build research query
    query = build_research_query(INPUT_FILE)
    print(f"✓ Research query built ({len(query)} chars)")

    # 4. Run deep research
    try:
        results = await run_deep_research(
            query=query,
            report_type="research_report"
        )
    except Exception as e:
        print(f"✗ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Save results
    save_results(results, RESEARCH_OUTPUT_FILE, SOURCES_FILE)

    # 6. Summary
    print("\n" + "=" * 60)
    print("Research Complete!")
    print("=" * 60)
    print(f"Report length: {len(results['report'])} characters")
    print(f"Sources found: {len(results['sources'])}")
    print(f"Images found: {len(results['images'])}")
    print(f"Costs: {results['costs']}")
    print(f"\nOutput files:")
    print(f"  - Report: {RESEARCH_OUTPUT_FILE}")
    print(f"  - Sources: {SOURCES_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
