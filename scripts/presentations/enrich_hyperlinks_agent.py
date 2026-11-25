#!/usr/bin/env python3
"""
Enrich slides with hyperlinks using an LLM agent (Pass 4.5).

Uses Claude to:
1. Identify all linkable terms in slides (tools, papers, frameworks, etc.)
2. Search our databases for matching URLs
3. Apply hyperlinks to the markdown

This is smarter than simple dictionary matching - it understands context
and can identify terms that should be linked even if not in a predefined list.

Output: 4.5_draft_slides_with_hyperlinks.md

Usage:
    python scripts/presentations/enrich_hyperlinks_agent.py --topic 01
    python scripts/presentations/enrich_hyperlinks_agent.py --all
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable

from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"

# Topic names
TOPIC_NAMES = {
    "01": "01_tools_and_open_source",
    "02": "02_genai_architectures",
    "03": "03_data_preparation",
    "04": "04_data_attribution",
    "05": "05_data_markets",
    "06": "06_visual_models",
    "07": "07_tabular_models",
    "08": "08_language_models",
    "09": "09_genai_in_science",
    "10": "10_serendipitous_findings",
    "11": "11_ethical_problems",
    "12": "12_commercial_applications",
    "13": "13_industry_insights",
    "14": "14_synthetic_data",
    "15": "15_whats_happening_at_neurips",
    "16": "16_neurips_awards",
    "17": "17_llm_scaling_laws_efficiency",
    "18": "18_agentic_ai_autonomous_systems",
    "19": "19_notable_quotes",
    "20": "20_small_language_models",
}

# Priority URLs - these are verified correct and override any database lookups
PRIORITY_URLS = {
    # Inference & Serving
    "vllm": "https://github.com/vllm-project/vllm",
    "vLLM": "https://github.com/vllm-project/vllm",
    "PagedAttention": "https://arxiv.org/abs/2309.06180",

    # Development Platforms
    "OpenHands": "https://github.com/All-Hands-AI/OpenHands",
    "SWE-bench": "https://github.com/princeton-nlp/SWE-bench",
    "SWE-Bench": "https://github.com/princeton-nlp/SWE-bench",
    "SWE-agent": "https://github.com/princeton-nlp/SWE-agent",
    "LiveCodeBench": "https://livecodebench.github.io/",
    "SWE-Bench Pro": "https://scale.com/blog/swe-bench-pro",

    # Training & RL
    "veRL": "https://github.com/volcengine/verl",
    "TRL": "https://github.com/huggingface/trl",
    "DeepSpeed": "https://github.com/microsoft/DeepSpeed",
    "Megatron-LM": "https://github.com/NVIDIA/Megatron-LM",

    # Frameworks
    "PyTorch": "https://pytorch.org",
    "JAX": "https://github.com/google/jax",
    "Hugging Face": "https://huggingface.co",
    "LangChain": "https://github.com/langchain-ai/langchain",
    "LlamaIndex": "https://github.com/run-llama/llama_index",

    # Models
    "Llama": "https://github.com/meta-llama/llama",
    "Llama 3": "https://github.com/meta-llama/llama3",
    "Mistral": "https://mistral.ai",
    "Claude": "https://anthropic.com",
    "GPT-4": "https://openai.com/gpt-4",
    "Gemini": "https://deepmind.google/technologies/gemini/",

    # Quantization
    "GPTQ": "https://github.com/IST-DASLab/gptq",
    "AWQ": "https://github.com/mit-han-lab/llm-awq",
    "llama.cpp": "https://github.com/ggerganov/llama.cpp",

    # Evaluation
    "HELM": "https://github.com/stanford-crfm/helm",
    "MLR-JUDGE": "https://arxiv.org/abs/2501.10018",

    # IBM / Generative Computing
    "Mellea": "https://research.ibm.com/blog/generative-computing-mellea",
    "Docling": "https://github.com/DS4SD/docling",
    "generative computing": "https://research.ibm.com/blog/generative-computing-mellea",
    "Generative Computing": "https://research.ibm.com/blog/generative-computing-mellea",

    # LLM360 / Open Training
    "LLM360": "https://www.llm360.ai/",
    "K2-V2": "https://huggingface.co/LLM360/K2-V2",

    # Apple / On-Device
    "MLX": "https://github.com/ml-explore/mlx",
    "Apple MLX": "https://github.com/ml-explore/mlx",
    "LM Studio": "https://lmstudio.ai/",

    # NVIDIA / Physical AI
    "NVIDIA Cosmos": "https://developer.nvidia.com/cosmos",
    "Cosmos": "https://developer.nvidia.com/cosmos",

    # Agent Frameworks
    "BeeAI": "https://github.com/i-am-bee/bee-agent-framework",
    "Aider": "https://github.com/paul-gauthier/aider",
    "Cursor": "https://cursor.sh",

    # Optimization
    "FlashAttention": "https://github.com/Dao-AILab/flash-attention",
    "Flash Attention": "https://github.com/Dao-AILab/flash-attention",
}

# Agent prompt for identifying linkable terms
IDENTIFY_LINKS_PROMPT = '''You are analyzing presentation slides to identify terms that should have hyperlinks.

SLIDES TO ANALYZE:
{slides_content}

AVAILABLE URL DATABASE:
{url_database}

YOUR TASK:
1. Read each bullet point and identify terms that should be hyperlinked
2. Match terms to URLs from the database when possible
3. For terms NOT in the database, suggest the most likely URL (GitHub, arXiv, official site)

WHAT TO LINK:
- AI/ML tools and frameworks (vLLM, PyTorch, LangChain, etc.)
- Research papers (by title or short name)
- Datasets and benchmarks (SWE-bench, HELM, etc.)
- Companies/organizations when relevant (OpenAI, Anthropic, etc.)
- Specific models (Llama, Mistral, Claude, etc.)
- Open source projects with GitHub repos

DO NOT LINK:
- Generic terms (AI, ML, deep learning, transformer, etc.)
- Terms that already have markdown links [text](url)
- Names of people
- Common concepts that don't have a specific URL

OUTPUT FORMAT (JSON):
{{
  "links_to_add": [
    {{
      "slide_num": 5,
      "original_text": "vLLM achieves 24x throughput",
      "term_to_link": "vLLM",
      "url": "https://github.com/vllm-project/vllm",
      "source": "database|inferred"
    }},
    {{
      "slide_num": 8,
      "original_text": "IBM's Mellea represents a contrarian view",
      "term_to_link": "Mellea",
      "url": "https://research.ibm.com/blog/generative-computing-mellea",
      "source": "database"
    }}
  ],
  "terms_without_urls": [
    {{
      "slide_num": 12,
      "term": "SomeObscureTool",
      "context": "mentioned in context of..."
    }}
  ]
}}

Be thorough but precise. Only link terms where you're confident about the URL.
'''

PAPER_URL_FIELDS = (
    "url",
    "arxiv_url",
    "openreview_url",
    "forum_url",
    "pdf_url",
    "github_url",
)


def _paper_title(paper: dict) -> str:
    for field in ("title", "paper_title", "source_title"):
        value = paper.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _paper_primary_url(paper: dict) -> str:
    for field in PAPER_URL_FIELDS:
        value = paper.get(field)
        if isinstance(value, str) and value.startswith("http"):
            return value
    return ""


def _iter_papers_from_json(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    papers = data.get("papers", [])
    return [paper for paper in papers if isinstance(paper, dict)]


def _add_paper_urls(url_db: dict, papers: Iterable[dict]) -> None:
    for paper in papers:
        title = _paper_title(paper)
        url = _paper_primary_url(paper)
        if not title or not url:
            continue
        entry = {
            "type": "paper",
            "arxiv_url": paper.get("arxiv_url") or url,
            "github_url": paper.get("github_url"),
            "url": url,
        }
        url_db[title.lower()] = entry
        words = title.split()
        if len(words) > 2:
            short = ' '.join(words[:3]).lower()
            url_db.setdefault(short, entry)


def load_url_database(topic_dir: Path, extra_paper_paths: Iterable[Path] | None = None) -> dict:
    """Load all available URL sources into a searchable database."""
    url_db = {}

    # Load ranked papers
    ranked_path = topic_dir / "ranked_papers.json"
    _add_paper_urls(url_db, _iter_papers_from_json(ranked_path))
    for paper_path in extra_paper_paths or []:
        _add_paper_urls(url_db, _iter_papers_from_json(Path(paper_path)))

    # Load source attributions
    attr_path = topic_dir / "source_attributions.json"
    if attr_path.exists():
        with open(attr_path) as f:
            data = json.load(f)
            for attr in data.get('attributions', []):
                title = attr.get('title', '')
                url = attr.get('neurips_url') or attr.get('url') or attr.get('openreview_url') or attr.get('forum_url') or ''
                if title and url:
                    url_db[title.lower()] = {
                        'type': 'neurips',
                        'url': url,
                    }

    # Load from analysis document (extract URLs)
    analysis_path = PROJECT_ROOT / "conferences" / "neurips2025" / "analysis"
    workspace_analysis = topic_dir.parents[1] / "analysis" if len(topic_dir.parents) > 1 else None
    if workspace_analysis and workspace_analysis.is_dir():
        analysis_path = workspace_analysis
    topic_num = topic_dir.name[:2]
    analysis_files = list(analysis_path.glob(f"{topic_num}_*.md"))
    for analysis_file in analysis_files:
        content = analysis_file.read_text()
        # Extract markdown links
        for match in re.finditer(r'\[([^\]]+)\]\((https?://[^)]+)\)', content):
            text, url = match.groups()
            if text.lower() not in url_db:
                url_db[text.lower()] = {
                    'type': 'analysis',
                    'url': url,
                }

    return url_db


def format_url_database_for_prompt(url_db: dict, max_entries: int = 200) -> str:
    """Format URL database for the agent prompt."""
    lines = []
    count = 0
    for name, info in sorted(url_db.items()):
        if count >= max_entries:
            break
        url = info.get('url') or info.get('arxiv_url') or info.get('github_url')
        if url:
            lines.append(f"- {name}: {url}")
            count += 1
    return '\n'.join(lines)


def format_slides_for_prompt(content: str) -> str:
    """Extract just the bullets from slides for the prompt."""
    lines = []
    slide_pattern = r'## Slide (\d+): ([^\n]+)\n(.*?)(?=## Slide \d+:|$)'

    for match in re.finditer(slide_pattern, content, re.DOTALL):
        slide_num = match.group(1)
        title = match.group(2)
        body = match.group(3)

        # Skip if already has many links
        if body.count('](http') > 3:
            continue

        # Extract bullets
        bullets = re.findall(r'^- (.+)$', body, re.MULTILINE)
        if bullets:
            lines.append(f"=== SLIDE {slide_num}: {title} ===")
            for bullet in bullets:
                # Skip bullets that already have links
                if '](http' not in bullet:
                    lines.append(f"  - {bullet}")
            lines.append("")

    return '\n'.join(lines)


def call_agent(prompt: str, verbose: bool = False) -> dict:
    """Call Claude API with the agent prompt."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed")
        return {}

    client = anthropic.Anthropic()

    if verbose:
        print("  Calling Claude to identify linkable terms...")

    try:
        message = anthropic_messages_create(client, caller=__file__,
            model=default_model_for_provider("presentation.hyperlink_enrichment", "anthropic"),
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        if verbose:
            print(f"    Agent error: {e}")
        return {}


def get_best_url(term: str, agent_url: str) -> str:
    """Get the best URL for a term, preferring priority URLs over agent suggestions."""
    # Check priority URLs first (case-insensitive)
    for key, url in PRIORITY_URLS.items():
        if key.lower() == term.lower():
            return url

    # Check if agent URL looks valid (not OpenAlex search, not generic)
    if agent_url:
        # Reject OpenAlex search URLs
        if 'openalex.org/works?search=' in agent_url:
            return ''
        # Reject generic search URLs
        if 'search=' in agent_url and ('google' in agent_url or 'bing' in agent_url):
            return ''
        return agent_url

    return ''


def apply_links(content: str, links_to_add: list[dict]) -> tuple[str, int]:
    """Apply hyperlinks to the markdown content."""
    links_applied = 0

    # Group links by slide for efficiency
    links_by_slide = {}
    for link in links_to_add:
        slide_num = link.get('slide_num')
        if slide_num not in links_by_slide:
            links_by_slide[slide_num] = []
        links_by_slide[slide_num].append(link)

    # Process each slide
    slide_pattern = r'(## Slide (\d+): [^\n]+\n)(.*?)(?=## Slide \d+:|$)'

    def replace_slide(match):
        nonlocal links_applied
        header = match.group(1)
        slide_num = int(match.group(2))
        body = match.group(3)

        if slide_num not in links_by_slide:
            return header + body

        # Apply each link for this slide
        for link in links_by_slide[slide_num]:
            term = link.get('term_to_link', '')
            agent_url = link.get('url', '')

            if not term:
                continue

            # Get best URL (priority > agent suggestion)
            url = get_best_url(term, agent_url)
            if not url or not url.startswith('http'):
                continue

            # Skip if already linked
            if f'[{term}](' in body or f'[{term.lower()}](' in body.lower():
                continue

            # Use word boundary matching
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, body, re.IGNORECASE):
                # Find the actual text to preserve case
                actual_match = re.search(pattern, body, re.IGNORECASE)
                if actual_match:
                    actual_term = actual_match.group()
                    # Only replace first occurrence
                    body = body[:actual_match.start()] + f'[{actual_term}]({url})' + body[actual_match.end():]
                    links_applied += 1

        return header + body

    result = re.sub(slide_pattern, replace_slide, content, flags=re.DOTALL)
    return result, links_applied


BATCH_SIZE = 10  # Process slides in batches to avoid output limits


def _split_slides_for_batching(content: str) -> list[str]:
    """Split formatted slide text into batches of BATCH_SIZE slides."""
    lines = content.split('\n')
    batches = []
    current_batch = []
    slide_count = 0

    for line in lines:
        if line.startswith('=== SLIDE '):
            if slide_count > 0 and slide_count % BATCH_SIZE == 0:
                batches.append('\n'.join(current_batch))
                current_batch = []
            slide_count += 1
        current_batch.append(line)

    if current_batch:
        batches.append('\n'.join(current_batch))

    return batches


def process_topic(topic_num: str, verbose: bool = False) -> Path:
    """Process a single topic: add hyperlinks to slides using LLM agent."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    # Prefer reading from existing 4.5 output (chain on top of dictionary pass)
    input_path = topic_dir / "4.5_draft_slides_with_hyperlinks.md"
    if not input_path.exists():
        input_path = topic_dir / "4_draft_slides_sanity_check.md"
    if not input_path.exists():
        input_path = topic_dir / "3_draft_slides_reflowed.md"
    if not input_path.exists():
        input_path = topic_dir / "2_draft_slides_enriched_with_papers.md"
    if not input_path.exists():
        input_path = topic_dir / "1_draft_slides.md"
    if not input_path.exists():
        raise FileNotFoundError(f"No input slides found in {topic_dir}")

    if verbose:
        print(f"\nEnriching hyperlinks (agent): {topic_name}")
        print(f"  Input: {input_path.name}")

    # Load URL database
    url_db = load_url_database(topic_dir)

    if verbose:
        print(f"  URL database entries: {len(url_db)}")

    # Read input markdown
    content = input_path.read_text()

    # Format for agent and split into batches
    slides_for_prompt = format_slides_for_prompt(content)
    url_db_for_prompt = format_url_database_for_prompt(url_db)
    batches = _split_slides_for_batching(slides_for_prompt)

    if verbose:
        print(f"  Processing {len(batches)} batch(es) of ~{BATCH_SIZE} slides each")

    # Call agent for each batch and collect results
    all_links = []
    all_terms_without_urls = []

    for i, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {i+1}/{len(batches)}...")

        prompt = IDENTIFY_LINKS_PROMPT.format(
            slides_content=batch,
            url_database=url_db_for_prompt
        )

        result = call_agent(prompt, verbose)
        batch_links = result.get('links_to_add', [])
        batch_terms = result.get('terms_without_urls', [])
        all_links.extend(batch_links)
        all_terms_without_urls.extend(batch_terms)

        if verbose:
            print(f"    Links identified: {len(batch_links)}")

    if verbose:
        print(f"  Total links identified: {len(all_links)}")
        print(f"  Terms without URLs: {len(all_terms_without_urls)}")

    # Apply links to content
    enriched_content, links_applied = apply_links(content, all_links)

    # Update metadata
    enriched_content = re.sub(
        r'GENERATOR: [^\n]+',
        'GENERATOR: enrich_hyperlinks_agent.py (Pass 4.5)',
        enriched_content
    )

    # Add timestamp and stats
    enriched_content += f"\n<!-- Hyperlinks enriched (agent): {datetime.now().isoformat()} -->\n"
    enriched_content += f"<!-- Links applied: {links_applied}, batches: {len(batches)} -->\n"

    # Save output
    output_path = topic_dir / "4.5_draft_slides_with_hyperlinks.md"
    output_path.write_text(enriched_content)

    if verbose:
        print(f"  Links applied: {links_applied}")
        print(f"  Saved: {output_path.name}")

        # Show some examples
        if all_links[:5]:
            print(f"  Sample links:")
            for link in all_links[:5]:
                print(f"    - {link.get('term_to_link')}: {link.get('url', '')[:50]}...")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Enrich slides with hyperlinks using LLM agent (Pass 4.5)"
    )
    parser.add_argument(
        "--topic", type=str,
        help="Topic number (01-20)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all topics"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )

    args = parser.parse_args()

    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root); this
    # legacy tool is hardcoded to the neurips2025 workspace (PRESENTATIONS_DIR).
    os.environ.setdefault(
        "PROVIDER_RECEIPT_WORKSPACE", str(PROJECT_ROOT / "conferences" / "neurips2025")
    )
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "enrich_hyperlinks_agent")

    if args.all:
        print("Enriching hyperlinks (agent) for all topics...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, args.verbose)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, verbose=True)
        print(f"Generated: {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
