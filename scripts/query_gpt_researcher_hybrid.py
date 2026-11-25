#!/usr/bin/env python3
"""
Query using GPT Researcher with Hybrid Search (Tavily Web + ChromaDB Local)

This script combines:
- Tavily web search for real-time internet results
- ChromaDB vector store for local NeurIPS presentation data
- GPT Researcher for comprehensive report generation

All results (report, context, sources) are saved for inspection.

CRITICAL: This script enforces reference quality requirements:
- Every finding must include: presentation name, ID, transcript path, timestamp, slide number
- Verbatim quotes are captured for verification
- Web findings are traced back to local data when possible
"""

# Reference quality prompt - instructs GPT-4o to include detailed citations
REFERENCE_QUALITY_PROMPT = """
CRITICAL INSTRUCTION FOR CITATIONS:

When writing this report, you MUST include precise, traceable references for every claim.
For each finding, include:

1. **Source Details Block** with:
   - Presentation: [Exact presentation title]
   - Presentation ID: [ID from metadata, e.g., deep_learning_code_agentic]
   - Timestamp: [HH:MM:SS - HH:MM:SS] when discussed
   - Slide: [slide number if from slides]
   - File paths when available

2. **Verbatim Quote** - Include the exact quote from the source in a blockquote:
   > "The exact words from the transcript or slide"

3. **Source Type** - Mark each source as:
   - [LOCAL] - From NeurIPS vector store (has transcript/slide reference)
   - [WEB] - From Tavily search (needs local traceability check)

Example format for each finding:

### [Tool/Concept Name]

**Summary:** [Description]

**Source Details:**
- Presentation: "Deep Learning for Code in the Agentic Era"
- Presentation ID: deep_learning_code_agentic
- Timestamp: 43:39-47:17
- Type: [LOCAL]

**Verbatim Quote:**
> "OpenHands is an open-source AI software engineer that can..."

If information comes from web search only, note:
- Source: [WEB - needs local verification]
- URL: [the source URL]
- Action: Check if this content exists in local NeurIPS data

DO NOT make claims without citing the specific source with these details.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add vector store config to path
vector_store_config = Path(__file__).parent.parent / 'conferences' / 'neurips2025' / 'vector_store' / 'gpt_researcher_config'
sys.path.insert(0, str(vector_store_config))


def load_env_file():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / ".env"
    env_vars = {}

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    # Handle 'export VAR=value' format
                    if line.startswith('export '):
                        line = line[7:]
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value

    return env_vars


def load_env_from_environment():
    """Return the API keys set in the environment.

    Keys are never read from a shell profile: source the repository's .env first
    (set -a; source .env; set +a).
    """
    return {key: value for key, value in os.environ.items() if key.endswith("_API_KEY")}


def setup_environment():
    """Setup environment variables from .env and the environment"""
    # First try .env file
    env_vars = load_env_file()

    # Fallback to variables already in the environment
    environment_vars = load_env_from_environment()

    # Merge (env file takes priority)
    for key, value in environment_vars.items():
        if key not in env_vars:
            env_vars[key] = value

    # Set environment variables
    required_keys = ['OPENAI_API_KEY', 'TAVILY_API_KEY']
    for key in required_keys:
        if key in env_vars:
            os.environ[key] = env_vars[key]
        elif key not in os.environ:
            print(f"Warning: {key} not found in .env or the environment")

    return env_vars


async def research_hybrid(
    query: str,
    report_type: str = "research_report",
    use_web: bool = True,
    use_local: bool = True,
    max_results: int = 15
):
    """
    Conduct hybrid research using both Tavily web search and ChromaDB local store.

    Args:
        query: Research question
        report_type: Type of report to generate
        use_web: Whether to include Tavily web search results
        use_local: Whether to include local ChromaDB results
        max_results: Maximum results from each source
    """
    try:
        from gpt_researcher import GPTResearcher
    except ImportError:
        print("Error: gpt-researcher not installed. Run: pip install gpt-researcher")
        sys.exit(1)

    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    vector_store_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store'
    chroma_dir = vector_store_dir / 'chroma_db'
    reports_dir = vector_store_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)

    # Check if vector store exists
    if use_local and not chroma_dir.exists():
        print("Warning: ChromaDB vector store not found!")
        print(f"Expected location: {chroma_dir}")
        print("Falling back to web search only...")
        use_local = False

    print(f"Query: {query}")
    print(f"Report type: {report_type}")
    print(f"Web search (Tavily): {'Enabled' if use_web else 'Disabled'}")
    print(f"Local search (ChromaDB): {'Enabled' if use_local else 'Disabled'}")
    print("-" * 60)

    all_context = []
    all_sources = []

    # 1. Query local ChromaDB first
    if use_local:
        print("\n[1/3] Searching local NeurIPS vector store...")
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_collection("neurips_presentations")

            results = collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )

            if results['documents'] and results['documents'][0]:
                print(f"    Found {len(results['documents'][0])} relevant chunks from NeurIPS data")

                for i, (doc, meta, dist) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Capture ALL available metadata for traceability
                    source_info = {
                        'source': 'local_chromadb',
                        'title': meta.get('title', 'Unknown'),
                        'presentation_id': meta.get('presentation_id', ''),
                        'paper_id': meta.get('paper_id', ''),
                        'type': meta.get('type', meta.get('source_type', 'unknown')),
                        'chunk_type': meta.get('chunk_type', ''),
                        'timestamp': meta.get('timestamp_range', meta.get('timestamp_formatted', meta.get('timestamp', ''))),
                        'timestamp_start': meta.get('timestamp_start', ''),
                        'timestamp_end': meta.get('timestamp_end', ''),
                        'slide_number': meta.get('slide_index', meta.get('slide_number', '')),
                        'slide_id': meta.get('slide_id', ''),
                        'url': meta.get('url', ''),
                        'distance': dist,
                        'similarity_score': 1 - dist,
                        'relevance': f"{(1 - dist) * 100:.1f}%",
                        'verbatim_excerpt': doc[:500] + '...' if len(doc) > 500 else doc,
                        'metadata': meta,
                    }
                    all_sources.append(source_info)

                    # Format context WITH detailed metadata for GPT Researcher
                    # This helps GPT-4o include proper citations
                    meta_block = f"""
[SOURCE: LOCAL - NeurIPS 2025]
Presentation: "{meta.get('title', 'Unknown')}"
Presentation ID: {meta.get('presentation_id', 'unknown')}
Chunk Type: {meta.get('chunk_type', 'unknown')}
Timestamp: {meta.get('timestamp', meta.get('timestamp_start', 'N/A'))}
Slide: {meta.get('slide_number', 'N/A')}
Relevance: {(1 - dist) * 100:.1f}%

CONTENT:
{doc}
"""
                    all_context.append(meta_block)
        except Exception as e:
            print(f"    Warning: ChromaDB query failed: {e}")

    # 2. Query Tavily for web results
    if use_web:
        print("\n[2/3] Searching web with Tavily...")
        try:
            from tavily import TavilyClient

            tavily_key = os.environ.get('TAVILY_API_KEY')
            if not tavily_key:
                print("    Warning: TAVILY_API_KEY not set, skipping web search")
            else:
                client = TavilyClient(api_key=tavily_key)

                # Search with NeurIPS context
                search_query = f"NeurIPS 2025 {query}"
                response = client.search(
                    query=search_query,
                    search_depth="advanced",
                    max_results=max_results,
                    include_answer=True,
                    include_raw_content=False
                )

                if response.get('results'):
                    print(f"    Found {len(response['results'])} web results")

                    for result in response['results']:
                        source_info = {
                            'source': 'tavily_web',
                            'title': result.get('title', 'Unknown'),
                            'url': result.get('url', ''),
                            'score': result.get('score', 0),
                            'local_match': None,  # Will be filled by traceability check
                            'needs_crawl': True,  # Assume needs crawl until verified
                        }
                        all_sources.append(source_info)

                        # Format context with WEB marker for GPT Researcher
                        content = result.get('content', '')
                        context_entry = f"""
[SOURCE: WEB - Needs Local Verification]
Title: "{result.get('title', 'Unknown')}"
URL: {result.get('url', '')}
Tavily Score: {result.get('score', 0):.2f}
NOTE: This is from web search. Check if content exists in local NeurIPS data.

CONTENT:
{content}
"""
                        all_context.append(context_entry)

                # Include Tavily's generated answer if available
                if response.get('answer'):
                    all_context.insert(0, f"[Tavily Summary]\n{response['answer']}")

        except ImportError:
            print("    Warning: tavily-python not installed. Run: pip install tavily-python")
        except Exception as e:
            print(f"    Warning: Tavily search failed: {e}")

    # 3. Generate report with GPT Researcher
    print("\n[3/3] Generating comprehensive report with GPT-4o...")

    # Combine all context with reference quality instructions
    combined_context = "\n\n---\n\n".join(all_context)

    # Prepend the reference quality instructions to ensure GPT-4o includes proper citations
    enhanced_context = f"""
{REFERENCE_QUALITY_PROMPT}

=== BEGIN SOURCE MATERIALS ===

{combined_context}

=== END SOURCE MATERIALS ===

Remember: Include Source Details blocks with Presentation ID, Timestamp, Slide number, and Verbatim Quotes for EVERY finding.
"""

    # Enhance the query to emphasize reference requirements
    enhanced_query = f"""{query}

IMPORTANT: For each finding, you MUST include:
1. Source Details (Presentation name, ID, Timestamp, Slide number)
2. Verbatim quotes from the source
3. Mark sources as [LOCAL] or [WEB - needs verification]
"""

    # Use GPT Researcher with the combined context
    # depth=5, breadth=8 (increased for thorough LLM exploration)
    researcher = GPTResearcher(
        query=enhanced_query,
        report_type=report_type,
        max_subtopics=8,  # breadth: number of sub-topics to explore
    )

    # Inject the enhanced context
    if enhanced_context:
        researcher.context = [enhanced_context]

    # Generate report
    report = await researcher.write_report()

    return report, all_context, all_sources


def check_web_traceability(sources: list, chroma_dir: Path) -> list:
    """
    Check if web sources can be traced to local data.

    For each web source, search ChromaDB for related content.
    Updates the sources list with local_match and needs_crawl fields.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("neurips_presentations")
    except Exception as e:
        print(f"    Warning: Could not check traceability: {e}")
        return sources

    print("\n[4/4] Checking web-to-local traceability...")

    web_sources = [s for s in sources if s['source'] == 'tavily_web']

    for src in web_sources:
        title = src.get('title', '')
        # Extract key terms from the title for search
        search_query = title[:100]  # Use first 100 chars of title

        try:
            results = collection.query(
                query_texts=[search_query],
                n_results=3,
                include=["metadatas", "distances"]
            )

            if results['distances'] and results['distances'][0]:
                best_distance = min(results['distances'][0])
                # If distance < 0.5, we likely have related local content
                if best_distance < 0.5:
                    src['local_match'] = f"✅ Found (dist: {best_distance:.2f})"
                    src['needs_crawl'] = False
                    matched_title = results['metadatas'][0][0].get('title', 'Unknown')
                    src['matched_presentation'] = matched_title
                else:
                    src['local_match'] = f"❌ No match (dist: {best_distance:.2f})"
                    src['needs_crawl'] = True
            else:
                src['local_match'] = "❌ No results"
                src['needs_crawl'] = True

        except Exception as e:
            src['local_match'] = f"⚠️ Error: {str(e)[:30]}"
            src['needs_crawl'] = True

    # Report summary
    matched = len([s for s in web_sources if not s.get('needs_crawl', True)])
    gaps = len([s for s in web_sources if s.get('needs_crawl', True)])
    print(f"    Web sources: {len(web_sources)} total")
    print(f"    ✅ Matched locally: {matched}")
    print(f"    🔴 Source gaps (need crawling): {gaps}")

    return sources


def save_results(query: str, report: str, context: list, sources: list, output_dir: Path):
    """Save all results for inspection"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query[:40].replace(" ", "_").replace("?", "").replace("/", "_").lower()

    # Create subdirectory for this query
    query_dir = output_dir / f"{timestamp}_{safe_query}"
    query_dir.mkdir(exist_ok=True)

    # 1. Save the report
    report_file = query_dir / "report.md"
    with open(report_file, 'w') as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Sources:** {len(sources)} total (local + web)\n\n")
        f.write(f"---\n\n")
        f.write(report)

    # 2. Save raw context
    context_file = query_dir / "raw_context.json"
    with open(context_file, 'w') as f:
        json.dump({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'context_chunks': context,
            'total_chunks': len(context)
        }, f, indent=2)

    # 3. Save sources list
    sources_file = query_dir / "sources.json"
    with open(sources_file, 'w') as f:
        json.dump({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': sources,
            'local_count': len([s for s in sources if s['source'] == 'local_chromadb']),
            'web_count': len([s for s in sources if s['source'] == 'tavily_web'])
        }, f, indent=2)

    # 4. Save sources as readable markdown with traceability info
    sources_md_file = query_dir / "sources.md"
    with open(sources_md_file, 'w') as f:
        f.write(f"# Sources for: {query}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Local Sources (NeurIPS Vector Store) ✅\n\n")
        f.write("These sources have full traceability to presentations, transcripts, and slides.\n\n")
        local_sources = [s for s in sources if s['source'] == 'local_chromadb']
        for i, src in enumerate(local_sources, 1):
            f.write(f"### {i}. {src['title']}\n\n")
            f.write(f"| Field | Value |\n")
            f.write(f"|-------|-------|\n")
            f.write(f"| Presentation ID | `{src.get('presentation_id', 'N/A')}` |\n")
            f.write(f"| Chunk Type | {src.get('chunk_type', 'N/A')} |\n")
            f.write(f"| Timestamp | {src.get('timestamp', src.get('timestamp_start', 'N/A'))} |\n")
            f.write(f"| Slide | {src.get('slide_number', 'N/A')} |\n")
            f.write(f"| Relevance | {src.get('relevance', 'N/A')} |\n")
            if src.get('url'):
                f.write(f"| URL | {src['url']} |\n")
            f.write(f"\n**Excerpt:**\n> {src.get('verbatim_excerpt', 'N/A')[:200]}...\n\n")
            f.write("---\n\n")

        f.write("\n## Web Sources (Tavily) ⚠️ NEEDS LOCAL VERIFICATION\n\n")
        f.write("These sources came from web search and need to be traced back to local data.\n\n")
        web_sources = [s for s in sources if s['source'] == 'tavily_web']

        if web_sources:
            f.write("### Source Gaps Table\n\n")
            f.write("| # | Title | URL | Local Match | Action |\n")
            f.write("|---|-------|-----|-------------|--------|\n")
            for i, src in enumerate(web_sources, 1):
                local_match = src.get('local_match', '❓ Not checked')
                needs_crawl = '🔴 Crawl needed' if src.get('needs_crawl', True) else '✅ Available'
                title = src.get('title', 'Unknown')[:40] + '...' if len(src.get('title', '')) > 40 else src.get('title', 'Unknown')
                f.write(f"| {i} | {title} | [Link]({src.get('url', '')}) | {local_match} | {needs_crawl} |\n")

            f.write("\n### Web Source Details\n\n")
            for i, src in enumerate(web_sources, 1):
                f.write(f"#### {i}. {src['title']}\n\n")
                f.write(f"- **URL:** {src.get('url', 'N/A')}\n")
                f.write(f"- **Tavily Score:** {src.get('score', 0):.2f}\n")
                f.write(f"- **Local Match:** {src.get('local_match', 'Not yet checked')}\n")
                f.write(f"- **Needs Crawl:** {'Yes' if src.get('needs_crawl', True) else 'No'}\n\n")
        else:
            f.write("No web sources used.\n\n")

        f.write("\n---\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Local sources:** {len(local_sources)} (fully traceable)\n")
        f.write(f"- **Web sources:** {len(web_sources)} (need verification)\n")
        f.write(f"- **Source gaps to address:** {len([s for s in web_sources if s.get('needs_crawl', True)])}\n")

    return query_dir


async def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='GPT Researcher with Hybrid Search')
    parser.add_argument('query', nargs='*', help='Research question')
    parser.add_argument('--no-web', action='store_true', help='Disable Tavily web search')
    parser.add_argument('--no-local', action='store_true', help='Disable local ChromaDB search')
    parser.add_argument('--max-results', type=int, default=25, help='Max results per source (increased for thorough LLM research)')

    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Get query
    if args.query:
        query = " ".join(args.query)
    else:
        print("Usage: python query_gpt_researcher_hybrid.py <your research question>")
        print()
        print("Options:")
        print("  --no-web      Disable Tavily web search (local only)")
        print("  --no-local    Disable ChromaDB search (web only)")
        print("  --max-results Maximum results per source (default: 15)")
        print()
        print("Example queries:")
        print('  "What new open source tools were announced at NeurIPS 2025?"')
        print('  "Compare different approaches to foundation models"')
        print('  "How is reinforcement learning applied to robotics?"')
        print()
        print("All results are saved to: conferences/neurips2025/cache/vector-store/reports/")
        sys.exit(1)

    print("=" * 60)
    print("GPT Researcher - Hybrid Search (Tavily + ChromaDB)")
    print("=" * 60)
    print()

    # Conduct research
    report, context, sources = await research_hybrid(
        query,
        use_web=not args.no_web,
        use_local=not args.no_local,
        max_results=args.max_results
    )

    # Check web-to-local traceability
    project_root = Path(__file__).parent.parent
    chroma_dir = project_root / 'conferences' / 'neurips2025' / 'vector_store' / 'chroma_db'
    if not args.no_web and chroma_dir.exists():
        sources = check_web_traceability(sources, chroma_dir)

    # Save results
    output_dir = Path(__file__).parent.parent / "conferences" / "neurips2025" / "vector_store" / "reports"
    output_dir.mkdir(exist_ok=True)

    query_dir = save_results(query, report, context, sources, output_dir)

    # Display results
    print()
    print("=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print()
    print(report)
    print()
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print(f"Directory: {query_dir}")
    print(f"  - report.md       : Final research report")
    print(f"  - sources.md      : Human-readable source list")
    print(f"  - sources.json    : Machine-readable sources")
    print(f"  - raw_context.json: All retrieved context chunks")
    print()


if __name__ == "__main__":
    asyncio.run(main())
