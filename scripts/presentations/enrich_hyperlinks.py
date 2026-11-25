#!/usr/bin/env python3
"""
Enrich slides with hyperlinks (Pass 4.5).

Reads 4_draft_slides_sanity_check.md and systematically adds hyperlinks to:
- Paper titles (arXiv URLs)
- GitHub repositories
- Workshops and presentations (NeurIPS virtual site URLs)
- Tools and frameworks with known URLs

Uses ranked_papers.json and source_attributions.json as URL databases.

Output: 4.5_draft_slides_with_hyperlinks.md

Usage:
    python scripts/presentations/enrich_hyperlinks.py --topic 01
    python scripts/presentations/enrich_hyperlinks.py --all
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections.abc import Iterable

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

# Well-known tools/frameworks with their URLs
KNOWN_TOOL_URLS = {
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
    "SWE-Bench-Live": "https://github.com/ofirpress/swe-bench-live",
    "SWE-Bench Pro": "https://scale.com/blog/swe-bench-pro",

    # Training & RL
    "veRL": "https://github.com/volcengine/verl",
    "TRL": "https://github.com/huggingface/trl",
    "DeepSpeed": "https://github.com/microsoft/DeepSpeed",
    "Megatron-LM": "https://github.com/NVIDIA/Megatron-LM",

    # Frameworks
    "PyTorch": "https://pytorch.org",
    "PyTorch 2": "https://pytorch.org/get-started/pytorch-2.0/",
    "JAX": "https://github.com/google/jax",
    "Hugging Face": "https://huggingface.co",
    "HuggingFace": "https://huggingface.co",
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
    "GGML": "https://github.com/ggerganov/ggml",
    "llama.cpp": "https://github.com/ggerganov/llama.cpp",

    # Evaluation
    "HELM": "https://github.com/stanford-crfm/helm",
    "lm-evaluation-harness": "https://github.com/EleutherAI/lm-evaluation-harness",
    "BigBench": "https://github.com/google/BIG-bench",
    "MLR-JUDGE": "https://arxiv.org/abs/2501.10018",
    "Eval360": "https://www.llm360.ai/",

    # Data
    "The Pile": "https://pile.eleuther.ai",
    "RedPajama": "https://github.com/togethercomputer/RedPajama-Data",
    "LAION": "https://laion.ai",

    # Scientific
    "AlphaFold": "https://github.com/google-deepmind/alphafold",
    "RoseTTAFold": "https://github.com/RosettaCommons/RoseTTAFold",

    # IBM / Generative Computing
    "Mellea": "https://research.ibm.com/blog/generative-computing-mellea",
    "Docling": "https://github.com/DS4SD/docling",
    "generative computing": "https://research.ibm.com/blog/generative-computing-mellea",

    # LLM360 / Open Training
    "LLM360": "https://www.llm360.ai/",
    "K2-V2": "https://huggingface.co/LLM360/K2-V2",
    "360-Open": "https://www.llm360.ai/",

    # Apple / On-Device
    "MLX": "https://github.com/ml-explore/mlx",
    "Apple MLX": "https://github.com/ml-explore/mlx",
    "LM Studio": "https://lmstudio.ai/",

    # NVIDIA / Physical AI
    "NVIDIA Cosmos": "https://developer.nvidia.com/cosmos",
    "Cosmos": "https://developer.nvidia.com/cosmos",
    "DRIVE Alpamayo": "https://developer.nvidia.com/drive",

    # Agent Frameworks
    "BeeAI": "https://github.com/i-am-bee/bee-agent-framework",
    "AgentBreeder": "https://arxiv.org/abs/2404.02403",
    "RepoMaster": "https://arxiv.org/abs/2501.02471",
    "ContextAgent": "https://arxiv.org/abs/2501.07653",

    # Code & Development
    "Aider": "https://github.com/paul-gauthier/aider",
    "Cursor": "https://cursor.sh",
    "Copilot": "https://github.com/features/copilot",
    "GitHub Copilot": "https://github.com/features/copilot",

    # Speech
    "Parakeet": "https://github.com/NVIDIA/NeMo",
    "Sortformer": "https://arxiv.org/abs/2409.06656",

    # Optimization
    "MPAX": "https://github.com/google/mpax",
    "Optax": "https://github.com/google-deepmind/optax",
    "FlashAttention": "https://github.com/Dao-AILab/flash-attention",
    "Flash Attention": "https://github.com/Dao-AILab/flash-attention",
}


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
            "arxiv_url": paper.get("arxiv_url") or url,
            "github_url": paper.get("github_url"),
            "url": url,
            "title": title,
        }
        url_db["papers"][title.lower()] = entry
        short_title = title[:30].lower()
        url_db["papers"].setdefault(short_title, entry)


def load_url_database(topic_dir: Path, extra_paper_paths: Iterable[Path] | None = None) -> dict:
    """Load URL mappings from ranked papers and source attributions."""
    url_db = {
        'papers': {},      # title -> {arxiv_url, github_url}
        'workshops': {},   # title -> neurips_url
        'presentations': {},  # title -> neurips_url
        'tools': KNOWN_TOOL_URLS.copy(),
    }

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
                category = attr.get('category', '')

                if title and url:
                    if category == 'Workshop':
                        url_db['workshops'][title.lower()] = {
                            'url': url,
                            'title': title,
                        }
                    elif category in ('Presentation', 'Keynote', 'Session'):
                        url_db['presentations'][title.lower()] = {
                            'url': url,
                            'title': title,
                        }

    return url_db


def _existing_link_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of existing markdown links in text."""
    spans = []
    for m in re.finditer(r'\[[^\]]+\]\([^)]+\)', text):
        spans.append((m.start(), m.end()))
    return spans


def _in_existing_link(pos: int, end: int, link_spans: list[tuple[int, int]]) -> bool:
    """Check if a position range overlaps with an existing markdown link."""
    for ls, le in link_spans:
        if pos < le and end > ls:
            return True
    return False


def find_linkable_terms(text: str, url_db: dict) -> list[dict]:
    """Find terms in text that can be hyperlinked."""
    matches = []
    text_lower = text.lower()

    # Track existing markdown link spans so we don't double-link
    link_spans = _existing_link_spans(text)

    # Check for paper titles (prioritize longer matches, allow multiple)
    for title_key, info in sorted(url_db['papers'].items(), key=lambda x: -len(x[0])):
        if title_key in text_lower:
            # Find the actual text position (case-insensitive)
            idx = text_lower.find(title_key)
            if idx != -1:
                original_text = text[idx:idx + len(title_key)]
                # Expand to full word boundaries if needed
                while idx > 0 and text[idx-1].isalnum():
                    idx -= 1
                end_idx = idx + len(title_key)
                while end_idx < len(text) and text[end_idx].isalnum():
                    end_idx += 1
                original_text = text[idx:end_idx]

                # Skip if inside an existing link or overlapping another match
                if _in_existing_link(idx, end_idx, link_spans):
                    continue
                overlaps = any(m['start'] <= idx < m['end'] or m['start'] < end_idx <= m['end']
                               for m in matches)
                if overlaps:
                    continue

                url = info.get('arxiv_url') or info.get('github_url')
                if url:
                    matches.append({
                        'start': idx,
                        'end': end_idx,
                        'text': original_text,
                        'url': url,
                        'type': 'paper',
                    })

    # Check for known tools
    for tool_name, url in url_db['tools'].items():
        # Use word boundary matching
        pattern = r'\b' + re.escape(tool_name) + r'\b'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Skip if inside an existing link or overlapping another match
            if _in_existing_link(match.start(), match.end(), link_spans):
                continue
            overlaps = any(m['start'] <= match.start() < m['end'] or m['start'] < match.end() <= m['end']
                          for m in matches)
            if not overlaps:
                matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'url': url,
                    'type': 'tool',
                })

    # Check for workshops
    for title_key, info in url_db['workshops'].items():
        # Use partial matching for workshop names
        words = title_key.split()[:4]  # First 4 words
        partial_key = ' '.join(words)
        if len(partial_key) > 10 and partial_key in text_lower:
            idx = text_lower.find(partial_key)
            if idx != -1:
                if _in_existing_link(idx, idx + len(partial_key), link_spans):
                    continue
                overlaps = any(m['start'] <= idx < m['end'] for m in matches)
                if not overlaps:
                    matches.append({
                        'start': idx,
                        'end': idx + len(partial_key),
                        'text': text[idx:idx + len(partial_key)],
                        'url': info['url'],
                        'type': 'workshop',
                    })

    # Sort by position
    matches.sort(key=lambda m: m['start'])

    return matches


def apply_hyperlinks(text: str, matches: list[dict]) -> str:
    """Apply hyperlinks to text based on matches."""
    if not matches:
        return text

    # Build result by replacing matches in reverse order (to preserve positions)
    result = text
    for match in reversed(matches):
        if match['url'] and match['url'].startswith('http'):
            link_text = match['text']
            link = f"[{link_text}]({match['url']})"
            result = result[:match['start']] + link + result[match['end']:]

    return result


def enrich_bullet(bullet: str, url_db: dict) -> tuple[str, int]:
    """Enrich a single bullet with hyperlinks. Returns (enriched_bullet, links_added)."""
    matches = find_linkable_terms(bullet, url_db)
    if matches:
        enriched = apply_hyperlinks(bullet, matches)
        return enriched, len(matches)

    return bullet, 0


def process_slide_content(content: str, url_db: dict) -> tuple[str, int]:
    """Process slide content and add hyperlinks to bullets."""
    lines = content.split('\n')
    result_lines = []
    total_links = 0

    for line in lines:
        if line.startswith('- '):
            bullet_text = line[2:]
            enriched, links = enrich_bullet(bullet_text, url_db)
            result_lines.append(f"- {enriched}")
            total_links += links
        else:
            result_lines.append(line)

    return '\n'.join(result_lines), total_links


def process_topic(topic_num: str, verbose: bool = False) -> Path:
    """Process a single topic: add hyperlinks to slides."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    # Try to load from Pass 4 (sanity check), fall back to earlier passes
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
        print(f"\nEnriching hyperlinks: {topic_name}")
        print(f"  Input: {input_path.name}")

    # Load URL database
    url_db = load_url_database(topic_dir)

    if verbose:
        print(f"  Papers indexed: {len(url_db['papers'])}")
        print(f"  Workshops indexed: {len(url_db['workshops'])}")
        print(f"  Presentations indexed: {len(url_db['presentations'])}")
        print(f"  Known tools: {len(url_db['tools'])}")

    # Read input markdown
    content = input_path.read_text()

    # Parse slides
    slide_pattern = r'(## Slide \d+: [^\n]+\n)(.*?)(?=## Slide \d+:|$)'

    total_links = 0
    slides_enriched = 0

    def process_match(match):
        nonlocal total_links, slides_enriched
        header = match.group(1)
        body = match.group(2)

        enriched_body, links = process_slide_content(body, url_db)
        if links > 0:
            total_links += links
            slides_enriched += 1

        return header + enriched_body

    # Process all slides
    enriched_content = re.sub(slide_pattern, process_match, content, flags=re.DOTALL)

    # Update metadata
    enriched_content = re.sub(
        r'GENERATOR: [^\n]+',
        'GENERATOR: enrich_hyperlinks.py (Pass 4.5)',
        enriched_content
    )

    # Add timestamp
    enriched_content += f"\n<!-- Hyperlinks enriched: {datetime.now().isoformat()} -->\n"

    # Save output
    output_path = topic_dir / "4.5_draft_slides_with_hyperlinks.md"
    output_path.write_text(enriched_content)

    if verbose:
        print(f"  Links added: {total_links}")
        print(f"  Slides enriched: {slides_enriched}")
        print(f"  Saved: {output_path.name}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Enrich slides with hyperlinks (Pass 4.5)"
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

    if args.all:
        print("Enriching hyperlinks for all topics...")
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
