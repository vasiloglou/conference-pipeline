"""
Claude Deep Research - Native Anthropic Implementation

This script implements deep research using Claude's native web search tool
and a multi-step research pattern inspired by Anthropic's multi-agent research system.

The approach:
1. Analyze the input document and generate research subtopics
2. For each subtopic, conduct focused web searches
3. Synthesize all findings into a comprehensive report with citations

Requirements:
    pip install anthropic python-dotenv

Environment Variables Required:
    ANTHROPIC_API_KEY - Your Anthropic API key

Note: Web search must be enabled in your Anthropic Console settings.
Pricing: See src/conference_pipeline/costs/pricing_catalog.json and provider receipts.

Usage:
    # Full research (default)
    python claude_deep_research.py

    # Re-attempt synthesis from saved findings
    python claude_deep_research.py --synthesis-only

    # Synthesis from specific findings file
    python claude_deep_research.py --synthesis-only --findings /path/to/raw_findings.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic import APIConnectionError, APITimeoutError, RateLimitError
from dotenv import load_dotenv

from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    infer_conference_workspace,
)

try:
    from conference_pipeline.costs.estimator import load_pricing_catalog
except ImportError:
    _repo_src = Path(__file__).resolve().parents[1] / "src"
    if _repo_src.exists() and str(_repo_src) not in sys.path:
        sys.path.insert(0, str(_repo_src))
    try:
        from conference_pipeline.costs.estimator import load_pricing_catalog
    except ImportError:
        load_pricing_catalog = None

from conference_pipeline.model_governance.registry import default_model_for_provider

# Load environment variables from .env file if present
load_dotenv()


def api_call_with_retry(func, max_retries: int = 5, initial_delay: float = 5.0):
    """
    Wrapper to retry API calls with exponential backoff.
    Handles connection errors, timeouts, and rate limits.
    """
    import time

    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"      API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                print(f"      Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"      API error after {max_retries} attempts: {e}")

    raise last_exception

# Configuration
BASE_DIR = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
INPUT_FILE = BASE_DIR / "analysis" / "20_small_language_models.md"
OUTPUT_DIR = BASE_DIR / "experimental_videos" / "20_small_language_models" / "claude-deep-research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_OUTPUT_FILE = OUTPUT_DIR / "claude_research_result.md"
SOURCES_FILE = OUTPUT_DIR / "research_sources.md"

# Model configuration: governed by the model registry, never literals.
RESEARCH_MODEL = default_model_for_provider("research.claude_deep_research", "anthropic")
SYNTHESIS_MODEL = default_model_for_provider("research.claude_deep_synthesis", "anthropic")

# Web search tool definition
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 10,  # Max searches per API call
}


def setup_environment() -> anthropic.Anthropic:
    """Verify API key and return configured client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found.\n"
            "Set it as an environment variable or in a .env file.\n"
            "Get your API key from: https://console.anthropic.com/"
        )

    return anthropic.Anthropic(api_key=api_key)


def generate_research_plan(client: anthropic.Anthropic, document_content: str) -> list[dict]:
    """
    Use Claude to analyze the document and generate focused research subtopics.

    Returns a list of research tasks with queries and focus areas.
    """
    print("\n[Phase 1] Generating research plan...")

    response = api_call_with_retry(lambda: anthropic_messages_create(client, caller=__file__,
        model=RESEARCH_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Analyze this document about Small Language Models from NeurIPS 2025 and create a research plan.

Document:
---
{document_content}
---

Generate 5-7 specific research subtopics that would help verify, expand, and deepen the information in this document.

For each subtopic, provide:
1. A clear research question
2. Specific search queries to use (2-3 per subtopic)
3. What to look for (papers, benchmarks, implementations, etc.)

Output as JSON array:
```json
[
  {{
    "subtopic": "Topic name",
    "research_question": "What we want to find out",
    "search_queries": ["query 1", "query 2"],
    "focus": "What types of sources to prioritize"
  }}
]
```

Focus on:
- Verifying specific papers and claims mentioned
- Finding recent (2024-2025) breakthrough papers
- Identifying real-world deployment case studies
- Discovering benchmark comparisons
- Finding implementation details and code repositories"""
        }]
    ))

    # Extract JSON from response
    response_text = response.content[0].text
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)

    if json_match:
        try:
            research_plan = json.loads(json_match.group(1))
            print(f"   Generated {len(research_plan)} research subtopics")
            return research_plan
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole response as JSON
    try:
        research_plan = json.loads(response_text)
        print(f"   Generated {len(research_plan)} research subtopics")
        return research_plan
    except json.JSONDecodeError:
        print("   Warning: Could not parse research plan, using default")
        return [
            {
                "subtopic": "Small Language Models Overview",
                "research_question": "What are the latest developments in SLMs?",
                "search_queries": [
                    "small language models NeurIPS 2024 2025",
                    "efficient LLM deployment edge devices"
                ],
                "focus": "Recent papers and benchmarks"
            }
        ]


def conduct_subtopic_research(
    client: anthropic.Anthropic,
    subtopic: dict,
    context: str
) -> dict:
    """
    Conduct web search research on a specific subtopic.

    Returns research findings with sources and citations.
    """
    print(f"\n   Researching: {subtopic['subtopic']}")

    # Combine queries into a focused research prompt
    queries_text = "\n".join(f"- {q}" for q in subtopic.get("search_queries", []))

    prompt = f"""Research the following topic thoroughly using web search:

Topic: {subtopic['subtopic']}
Research Question: {subtopic['research_question']}
Focus: {subtopic.get('focus', 'Recent papers and implementations')}

Suggested search queries:
{queries_text}

Context from the source document:
{context[:2000]}...

Instructions:
1. Search for recent (2024-2025) papers, especially from NeurIPS, ICML, ACL, arXiv
2. Find specific paper titles, authors, and URLs when possible
3. Look for benchmark results and comparisons
4. Identify key findings and insights
5. Note any contradictions or updates to the source document

Provide a detailed research summary with all sources cited."""

    try:
        response = api_call_with_retry(lambda: anthropic_messages_create(client, caller=__file__,
            model=RESEARCH_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            tools=[WEB_SEARCH_TOOL]
        ))

        # Extract text content and citations
        findings = {
            "subtopic": subtopic["subtopic"],
            "research_question": subtopic["research_question"],
            "content": "",
            "sources": [],
            "search_count": 0
        }

        for block in response.content:
            if block.type == "text":
                findings["content"] += block.text
                # Extract citations if present
                if hasattr(block, "citations") and block.citations:
                    for citation in block.citations:
                        if hasattr(citation, "url"):
                            findings["sources"].append({
                                "url": citation.url,
                                "title": getattr(citation, "title", ""),
                                "cited_text": getattr(citation, "cited_text", "")
                            })
            elif block.type == "web_search_tool_result":
                # Count searches and extract source URLs
                if hasattr(block, "content"):
                    for result in block.content:
                        if hasattr(result, "url"):
                            findings["sources"].append({
                                "url": result.url,
                                "title": getattr(result, "title", ""),
                                "page_age": getattr(result, "page_age", "")
                            })

        # Get search count from usage
        if hasattr(response, "usage") and hasattr(response.usage, "server_tool_use"):
            findings["search_count"] = getattr(
                response.usage.server_tool_use,
                "web_search_requests",
                0
            )

        print(f"      - {findings['search_count']} searches, {len(findings['sources'])} sources found")
        return findings

    except Exception as e:
        print(f"      - Error: {e}")
        return {
            "subtopic": subtopic["subtopic"],
            "research_question": subtopic["research_question"],
            "content": f"Research failed: {str(e)}",
            "sources": [],
            "search_count": 0
        }


def synthesize_finding(
    client: anthropic.Anthropic,
    finding: dict
) -> str:
    """
    Synthesize a single research finding into a structured summary.
    This is Pass 1 of the multi-pass synthesis approach.
    """
    content = finding['content'][:4000]  # Limit content size
    sources_text = ""
    if finding.get('sources'):
        sources_text = "\n\nKey Sources:\n"
        for source in finding['sources'][:5]:
            sources_text += f"- [{source.get('title', 'Link')}]({source.get('url', '')})\n"

    prompt = f"""Analyze this research finding and create a structured summary.

Research Topic: {finding['subtopic']}
Research Question: {finding['research_question']}

Raw Finding:
{content}
{sources_text}

Create a concise summary (300-500 words) with:
1. **Key Finding**: What was discovered or verified (1-2 sentences)
2. **Verification Status**: Was the original claim verified, partially verified, or unverified?
3. **Important Details**: Key technical details, metrics, or papers found
4. **Sources**: List the most relevant URLs found

Be factual and specific. Include actual paper names, metrics, and URLs where available."""

    response = api_call_with_retry(lambda: anthropic_messages_create(client, caller=__file__,
        model=SYNTHESIS_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    ))

    return response.content[0].text


def synthesize_report(
    client: anthropic.Anthropic,
    document_content: str,
    research_findings: list[dict]
) -> str:
    """
    Synthesize all research findings into a comprehensive report.
    Uses a multi-pass approach to work within Opus API limits.

    Pass 1: Synthesize each finding individually
    Pass 2: Merge individual syntheses into final report
    """
    print("\n[Phase 3] Synthesizing final report (multi-pass)...")

    # Pass 1: Synthesize each finding individually
    print("   Pass 1: Synthesizing individual findings...")
    individual_syntheses = []

    for i, finding in enumerate(research_findings, 1):
        print(f"      [{i}/{len(research_findings)}] {finding['subtopic'][:50]}...")
        try:
            synthesis = synthesize_finding(client, finding)
            individual_syntheses.append({
                "subtopic": finding['subtopic'],
                "synthesis": synthesis,
                "sources": finding.get('sources', [])
            })
        except Exception as e:
            print(f"         Warning: Failed to synthesize ({e}), using raw content")
            individual_syntheses.append({
                "subtopic": finding['subtopic'],
                "synthesis": finding['content'][:2000],
                "sources": finding.get('sources', [])
            })

    # Pass 2: Merge into final report
    print("   Pass 2: Merging into final report...")

    # Build condensed findings for merge
    # Strip URLs from content to avoid API issues with large URL counts
    import re
    condensed_findings = ""
    for item in individual_syntheses:
        condensed_findings += f"\n\n## {item['subtopic']}\n\n"
        # Remove URLs from synthesis text (they cause API connection issues)
        clean_synthesis = re.sub(r'https?://[^\s\)\]]+', '[URL]', item['synthesis'])
        condensed_findings += clean_synthesis

    # Collect all sources, filtering out irrelevant domains
    # The web search sometimes returns garbage results from unrelated sites
    irrelevant_domains = {
        'relational.ai', 'relationalai', 'docs.relational.ai',
        'rel.relational.ai', 'trust.relational.ai', 'www.relational.ai'
    }

    all_sources = []
    for item in individual_syntheses:
        for source in item.get('sources', []):
            url = source.get('url', '')
            if not url:
                continue
            # Skip irrelevant domains
            from urllib.parse import urlparse
            try:
                domain = urlparse(url).netloc.lower()
                if any(irrel in domain for irrel in irrelevant_domains):
                    continue
            except:
                pass
            # Skip duplicates
            if url not in [s.get('url') for s in all_sources]:
                all_sources.append(source)

    # Build sources section (limit to 15 relevant sources)
    sources_section = ""
    if all_sources:
        sources_section = "\n\n## Relevant Sources Found\n\n"
        for source in all_sources[:15]:
            title = source.get('title', 'Link')
            url = source.get('url', '')
            sources_section += f"- [{title}]({url})\n"
    else:
        sources_section = "\n\n## Sources\n\n(No relevant external sources found - search returned mostly irrelevant results)\n"

    print(f"   Condensed findings: {len(condensed_findings)} chars")

    # Use a simplified prompt to avoid API connection issues
    # Large prompts + complex templates + high max_tokens can cause server disconnects
    merge_prompt = f"""Create a research report based on these findings.

# Research Topic: Small Language Models at NeurIPS 2025

## Document Context (summary)
{document_content[:1500]}

## Research Findings
{condensed_findings[:15000]}

{sources_section}

---

Write a comprehensive research report with:

1. **Executive Summary** - 2-3 paragraphs of key findings
2. **Claims Verification** - Table showing which claims were Verified, Unverified, or Partially Verified
3. **Key Insights** - Main discoveries and patterns
4. **Research Gaps** - What couldn't be verified or found
5. **Recommendations** - For researchers and practitioners

Format as clean Markdown. Be specific and factual."""

    print(f"   Merge prompt: {len(merge_prompt)} chars")

    # Use Sonnet for merge with conservative max_tokens
    response = api_call_with_retry(lambda: anthropic_messages_create(client, caller=__file__,
        model=RESEARCH_MODEL,
        max_tokens=4000,  # Lower to avoid connection issues
        messages=[{"role": "user", "content": merge_prompt}]
    ))

    return response.content[0].text


def extract_all_sources(research_findings: list[dict]) -> list[dict]:
    """Extract and deduplicate all sources from research findings."""
    seen_urls = set()
    all_sources = []

    for finding in research_findings:
        for source in finding.get("sources", []):
            url = source.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append(source)

    return all_sources


def load_findings_from_json(findings_file: Path) -> dict:
    """Load research findings from a JSON file."""
    if not findings_file.exists():
        raise FileNotFoundError(f"Findings file not found: {findings_file}")

    with open(findings_file, "r") as f:
        data = json.load(f)

    required_keys = ["research_findings", "total_searches", "sources_count"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Invalid findings file: missing '{key}' key")

    return data


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Deep Research - Native Anthropic Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full research pipeline
  python claude_deep_research.py

  # Re-attempt synthesis from saved findings
  python claude_deep_research.py --synthesis-only

  # Synthesis from specific findings file
  python claude_deep_research.py --synthesis-only --findings /path/to/raw_findings.json
        """
    )

    parser.add_argument(
        "--synthesis-only",
        action="store_true",
        help="Skip research phase and only run synthesis from saved findings"
    )

    parser.add_argument(
        "--findings",
        type=Path,
        default=None,
        help="Path to raw_findings.json file (default: auto-detect in output directory)"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help=f"Input document to research (default: {INPUT_FILE})"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})"
    )

    return parser.parse_args()


def save_results(
    report: str,
    sources: list[dict],
    research_findings: list[dict],
    report_file: Path,
    sources_file: Path
):
    """Save research results to files."""
    # Save main report
    with open(report_file, "w") as f:
        f.write(f"# Claude Deep Research Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(report)

    print(f"\nReport saved to: {report_file}")

    # Save sources
    with open(sources_file, "w") as f:
        f.write("# Research Sources\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"Total unique sources: {len(sources)}\n\n")

        # Group by subtopic
        f.write("## Sources by Research Area\n\n")
        for finding in research_findings:
            f.write(f"### {finding['subtopic']}\n\n")
            for source in finding.get("sources", [])[:10]:
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                if url:
                    f.write(f"- [{title}]({url})\n")
            f.write("\n")

        # Full deduplicated list
        f.write("## All Unique Sources\n\n")
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            f.write(f"{i}. [{title}]({url})\n")

    print(f"Sources saved to: {sources_file}")


def run_synthesis_only(args):
    """Run synthesis from saved findings."""
    print("=" * 60)
    print("Claude Deep Research - SYNTHESIS ONLY MODE")
    print("=" * 60)

    # Determine findings file path
    output_dir = args.output_dir
    if args.findings:
        findings_file = args.findings
    else:
        findings_file = output_dir / "raw_findings.json"

    # 1. Setup client
    try:
        client = setup_environment()
        print(f"Environment configured")
        print(f"  - Model: {SYNTHESIS_MODEL}")
    except ValueError as e:
        print(f"Environment setup failed:\n{e}")
        return

    # 2. Load findings
    print(f"\nLoading findings from: {findings_file}")
    try:
        data = load_findings_from_json(findings_file)
        research_findings = data["research_findings"]
        total_searches = data["total_searches"]
        print(f"   Loaded {len(research_findings)} research topics")
        print(f"   Original searches: {total_searches}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading findings: {e}")
        return

    # 3. Load input document for context
    input_file = args.input
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    print(f"Input file: {input_file}")

    with open(input_file, "r") as f:
        document_content = f.read()
    print(f"Document loaded ({len(document_content)} chars)")

    # 4. Run synthesis
    all_sources = extract_all_sources(research_findings)
    print(f"\nTotal sources available: {len(all_sources)}")

    try:
        report = synthesize_report(client, document_content, research_findings)
        print("\nSynthesis completed successfully!")
    except Exception as e:
        print(f"\nSynthesis failed: {e}")
        print("Creating fallback report from raw findings...")
        report = "# Research Findings (Raw - Synthesis Failed)\n\n"
        report += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        report += f"Total searches: {total_searches}\n"
        report += f"Total sources: {len(all_sources)}\n\n"
        for finding in research_findings:
            report += f"## {finding['subtopic']}\n\n"
            report += f"**Question:** {finding['research_question']}\n\n"
            report += finding['content'][:5000] + "\n\n"
            if finding['sources']:
                report += "**Sources:**\n"
                for source in finding['sources'][:5]:
                    report += f"- [{source.get('title', 'Link')}]({source.get('url', '')})\n"
            report += "\n---\n\n"

    # 5. Save results
    report_file = output_dir / "claude_research_result.md"
    sources_file = output_dir / "research_sources.md"
    save_results(report, all_sources, research_findings, report_file, sources_file)

    # 6. Summary
    print("\n" + "=" * 60)
    print("Synthesis Complete!")
    print("=" * 60)
    print(f"Report length: {len(report)} characters")
    print(f"Total sources: {len(all_sources)}")
    print(f"\nOutput files:")
    print(f"  - Report: {report_file}")
    print(f"  - Sources: {sources_file}")


def run_full_research(args):
    """Run the full research pipeline."""
    print("=" * 60)
    print("Claude Deep Research (Native Anthropic Implementation)")
    print("=" * 60)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup
    try:
        client = setup_environment()
        print(f"Environment configured")
        print(f"  - Model: {RESEARCH_MODEL}")
    except ValueError as e:
        print(f"Environment setup failed:\n{e}")
        return

    # 2. Read input document
    input_file = args.input
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    print(f"Input file: {input_file}")

    with open(input_file, "r") as f:
        document_content = f.read()
    print(f"Document loaded ({len(document_content)} chars)")

    # 3. Generate research plan
    research_plan = generate_research_plan(client, document_content)

    # 4. Conduct research on each subtopic
    print("\n[Phase 2] Conducting web search research...")
    research_findings = []
    total_searches = 0

    for i, subtopic in enumerate(research_plan, 1):
        print(f"\n   [{i}/{len(research_plan)}] {subtopic['subtopic']}")
        findings = conduct_subtopic_research(client, subtopic, document_content)
        research_findings.append(findings)
        total_searches += findings.get("search_count", 0)

    print(f"\n   Total web searches performed: {total_searches}")

    # 4b. Save intermediate findings (in case synthesis fails)
    all_sources = extract_all_sources(research_findings)
    intermediate_file = output_dir / "raw_findings.json"
    try:
        with open(intermediate_file, "w") as f:
            json.dump({
                "research_findings": research_findings,
                "total_searches": total_searches,
                "sources_count": len(all_sources)
            }, f, indent=2, default=str)
        print(f"   Intermediate findings saved to: {intermediate_file}")
    except Exception as e:
        print(f"   Warning: Could not save intermediate findings: {e}")

    # 5. Synthesize final report
    try:
        report = synthesize_report(client, document_content, research_findings)
    except Exception as e:
        print(f"\n   Synthesis failed: {e}")
        print("   Saving raw findings instead...")
        print("   TIP: Re-run with --synthesis-only to retry synthesis later")
        # Create a basic report from raw findings
        report = "# Research Findings (Raw - Synthesis Failed)\n\n"
        report += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        report += f"Total searches: {total_searches}\n"
        report += f"Total sources: {len(all_sources)}\n\n"
        for finding in research_findings:
            report += f"## {finding['subtopic']}\n\n"
            report += f"**Question:** {finding['research_question']}\n\n"
            report += finding['content'][:5000] + "\n\n"
            if finding['sources']:
                report += "**Sources:**\n"
                for source in finding['sources'][:5]:
                    report += f"- [{source.get('title', 'Link')}]({source.get('url', '')})\n"
            report += "\n---\n\n"

    # 6. Save results
    report_file = output_dir / "claude_research_result.md"
    sources_file = output_dir / "research_sources.md"
    save_results(report, all_sources, research_findings, report_file, sources_file)

    # 7. Summary
    print("\n" + "=" * 60)
    print("Research Complete!")
    print("=" * 60)
    print(f"Report length: {len(report)} characters")
    print(f"Total sources: {len(all_sources)}")
    print(f"Web searches: {total_searches}")
    print(f"\nOutput files:")
    print(f"  - Report: {report_file}")
    print(f"  - Sources: {sources_file}")
    print(f"  - Raw data: {intermediate_file}")
    if load_pricing_catalog is not None:
        catalog = load_pricing_catalog()
        search_cost, _, _ = catalog.estimate_credits(
            "anthropic", "web_search_20250305", total_searches
        )
        print(f"\nEstimated search cost: ${search_cost:.2f}")
    else:
        print("\nEstimated search cost: run scripts/estimate_pipeline_costs.py for registry pricing.")


def main():
    """Main entry point for Claude deep research."""
    args = parse_args()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from --output-dir to avoid falling back to Path.cwd() (repo root); this
    # tool's default output is a legacy experimental directory that is not
    # under conferences/<slug>, in which case no workspace can be inferred.
    inferred_workspace = infer_conference_workspace(args.output_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "claude_deep_research")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if args.synthesis_only:
        run_synthesis_only(args)
    else:
        run_full_research(args)


if __name__ == "__main__":
    main()
