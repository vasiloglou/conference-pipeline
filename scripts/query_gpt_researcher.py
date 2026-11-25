#!/usr/bin/env python3
"""
Query GPT Researcher with Conference Data

Example script showing how to use GPT Researcher to query the indexed
conference presentations.
"""

import asyncio
import sys
from pathlib import Path

# Import GPT Researcher
try:
    from gpt_researcher import GPTResearcher
except ImportError:
    print("Error: gpt-researcher not installed. Run: pip install gpt-researcher")
    sys.exit(1)


async def research_question(query: str, report_type: str = "research_report"):
    """
    Research a question using the indexed conference presentations.

    Args:
        query: The research question
        report_type: Type of report to generate
            - "research_report": Detailed research report
            - "outline_report": Outline format
            - "resource_report": List of resources
            - "custom_report": Custom format
    """
    print(f"Researching: {query}")
    print(f"Report type: {report_type}")
    print("-" * 60)

    # Path to config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "conferences" / "neurips2025" / "gpt_researcher" / "gpt_researcher_config.json"

    # Create researcher
    # Config includes depth=4, breadth=6 (increased by 2 from defaults)
    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        config_path=str(config_path),
        max_subtopics=6,  # breadth: number of sub-topics to explore
    )

    # Conduct research
    print("Conducting research...")
    await researcher.conduct_research()

    # Generate report
    print("Generating report...")
    report = await researcher.write_report()

    return report


async def main():
    """Main function with example queries"""

    # Example queries
    example_queries = [
        "What are the key innovations in autoregressive models presented at NeurIPS 2025?",
        "Summarize the main approaches to visual generation models discussed in the conference.",
        "What are the latest developments in reinforcement learning for robotics?",
        "What papers discuss scaling laws for large language models?",
    ]

    if len(sys.argv) > 1:
        # Use query from command line
        query = " ".join(sys.argv[1:])
    else:
        # Use first example query
        print("No query provided. Using example query.")
        print("Usage: python query_gpt_researcher.py <your research question>")
        print()
        print("Example queries:")
        for i, q in enumerate(example_queries, 1):
            print(f"  {i}. {q}")
        print()
        query = example_queries[0]

    # Conduct research
    report = await research_question(query)

    # Display results
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print()
    print(report)
    print()

    # Save to file
    output_dir = Path(__file__).parent.parent / "conferences" / "neurips2025" / "gpt_researcher" / "reports"
    output_dir.mkdir(exist_ok=True)

    # Create filename from query
    filename = query[:50].replace(" ", "_").replace("?", "").lower() + ".md"
    output_file = output_dir / filename

    with open(output_file, 'w') as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(f"---\n\n")
        f.write(report)

    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
