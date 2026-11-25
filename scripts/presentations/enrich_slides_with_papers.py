#!/usr/bin/env python3
"""
Enrich presentation slides with paper references (Pass 2).

Takes 1_draft_slides.md and enriches it with papers from:
- ranked_papers.json (20 curated influential papers)
- exhaustive_extractions/papers/by_topic/*.jsonl (NeurIPS papers)

Uses 5 parallel agents:
- Agent 1: Maps ranked papers to existing slides, suggests new slides
- Agents 2-5: Each reviews ~350 NeurIPS papers to find supporting content

Output: 2_draft_slides_enriched_with_papers.md

Usage:
    python scripts/presentations/enrich_slides_with_papers.py --topic 01
    python scripts/presentations/enrich_slides_with_papers.py --all
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    infer_conference_workspace,
)
from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_SOURCE_BINDING_MISSING,
    BLUEPRINT_SOURCE_UNRESOLVED,
    PresentationBlueprintValidationError,
    require_approved_blueprint,
    validate_blueprint_file,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
PAPERS_BY_TOPIC_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "exhaustive_extractions" / "papers" / "by_topic"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
ENRICHMENT_PROVIDER_REQUIREMENTS = ["env:ANTHROPIC_API_KEY"]
PAPER_URL_FIELDS = (
    "url",
    "arxiv_url",
    "openreview_url",
    "forum_url",
    "pdf_url",
)

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

# Agent prompts
RANKED_PAPERS_MAPPER_PROMPT = '''You are enriching a presentation with paper references.

CURRENT SLIDES:
{draft_slides}

RANKED INFLUENTIAL PAPERS (20 papers, ordered by importance):
{ranked_papers}

YOUR TASK:
1. For each paper in the ranked list, determine which slide(s) it supports
2. Identify important papers (rank 1-10) that aren't covered by any existing slide
3. For uncovered top papers, propose NEW slides (aim for 5-10 new slides to reach 60 total)

RULES:
- Each new slide should have 3-5 bullet points
- Focus on concrete contributions, not generic descriptions
- Include paper rank in the SOURCE comment
- Prioritize papers with PDFs available (pdf_exists: true)
- IMPORTANT: When referencing a paper, use markdown hyperlinks: [Paper Name](arxiv_url)
- If GitHub URL is available, include it: [Paper Name](arxiv_url) | [GitHub](github_url)

OUTPUT FORMAT (JSON):
{{
  "paper_mappings": [
    {{
      "paper_rank": 1,
      "paper_title": "Paper title",
      "mapped_to_slides": [3, 4],
      "suggested_addition": "Add bullet: '[vLLM](https://arxiv.org/abs/2309.06180) achieves 24x throughput via PagedAttention'"
    }}
  ],
  "new_slides": [
    {{
      "title": "Slide Title (max 60 chars)",
      "based_on_papers": [3, 5],
      "bullets": [
        "[SWE-bench](https://arxiv.org/abs/...) enables real-world code evaluation",
        "Achieves 45% resolution rate on verified issues",
        "Third point on practical impact"
      ],
      "figure_need": "architecture diagram|benchmark chart|workflow diagram|none",
      "source": "Paper #3: SWE-bench"
    }}
  ],
  "coverage_summary": "Brief summary of what's well-covered vs gaps"
}}
'''

NEURIPS_PAPER_DISCOVERER_PROMPT = '''You are finding NeurIPS papers to enrich a presentation.

CURRENT SLIDE TITLES AND THEMES:
{slide_summaries}

NEURIPS PAPERS TO REVIEW (your assigned batch):
{papers_batch}

YOUR TASK:
1. Find papers that STRONGLY support existing slide themes
2. Identify papers that could create NEW slides for gaps
3. Be STRICT - only suggest highly relevant papers

SELECTION CRITERIA:
- Paper must directly relate to a slide theme OR fill an obvious gap
- Prefer papers with clear, concrete contributions
- Maximum 10 paper suggestions from your batch
- When URLs are provided, use markdown hyperlinks: [Paper Name](url)

OUTPUT FORMAT (JSON):
{{
  "supporting_papers": [
    {{
      "source_id": "paper_id",
      "title": "Paper Title",
      "supports_slide_num": 12,
      "suggested_bullet": "Concrete contribution to add (use [Paper Name](url) if URL provided)"
    }}
  ],
  "new_slide_proposals": [
    {{
      "source_id": "paper_id",
      "title": "Paper Title",
      "proposed_slide_title": "New Slide Title",
      "bullets": ["Point 1", "Point 2", "Point 3"],
      "fills_gap": "Description of what gap this fills"
    }}
  ]
}}
'''


def load_draft_slides(topic_dir: Path) -> str:
    """Load the base draft slides (Pass 1 output)."""
    draft_path = topic_dir / "1_draft_slides.md"
    if not draft_path.exists():
        raise FileNotFoundError(f"No 1_draft_slides.md found in {topic_dir}. Run generate_slides_simple.py first.")
    return draft_path.read_text()


def load_ranked_papers(topic_dir: Path) -> list[dict]:
    """Load canonical matched papers first, then legacy ranked papers."""
    papers_path = topic_dir / "matched_papers.json"
    if not papers_path.exists():
        papers_path = topic_dir / "ranked_papers.json"
    if not papers_path.exists():
        return []
    with open(papers_path) as f:
        data = json.load(f)
    return data.get('papers', [])


def load_neurips_papers(topic_num: str) -> list[dict]:
    """Load NeurIPS papers from by_topic JSONL."""
    jsonl_path = PAPERS_BY_TOPIC_DIR / f"{topic_num}.jsonl"
    if not jsonl_path.exists():
        return []

    papers = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return papers


def partition_papers(papers: list[dict], n_partitions: int = 4) -> list[list[dict]]:
    """Split papers into n partitions for parallel agents."""
    if not papers:
        return [[] for _ in range(n_partitions)]

    partition_size = len(papers) // n_partitions
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size if i < n_partitions - 1 else len(papers)
        partitions.append(papers[start:end])
    return partitions


def extract_slide_summaries(draft_slides: str) -> str:
    """Extract slide titles and key themes for agent context."""
    summaries = []
    slide_pattern = r'## Slide (\d+): (.+?)(?:\n|$)'
    for match in re.finditer(slide_pattern, draft_slides):
        num = match.group(1)
        title = match.group(2)
        summaries.append(f"Slide {num}: {title}")
    return "\n".join(summaries)


def _paper_primary_url(paper: dict[str, Any]) -> str:
    for field in PAPER_URL_FIELDS:
        value = paper.get(field)
        if isinstance(value, str) and value.startswith("http"):
            return value
    return ""


def _paper_title(paper: dict[str, Any]) -> str:
    for field in ("title", "paper_title", "source_title"):
        value = paper.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _paper_url_index(papers: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for paper in papers:
        title = _paper_title(paper)
        url = _paper_primary_url(paper)
        if not title or not url:
            continue
        normalized = _normalize_link_text(title)
        index[normalized] = url
        words = normalized.split()
        for length in range(min(6, len(words)), 1, -1):
            index.setdefault(" ".join(words[:length]), url)
    return index


def _normalize_link_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def replace_unresolved_paper_url_placeholders(
    markdown: str,
    papers: list[dict[str, Any]],
) -> str:
    """Replace LLM-produced ``paper_url`` placeholders with known paper URLs.

    The enrichment prompt asks the model to use markdown links when a URL is
    provided. Some providers still emit ``[Title](paper_url)``. Keeping that
    placeholder makes the deck look linked while actually losing the source, so
    resolve it from the matched-paper URL index or fall back to plain text.
    """

    url_index = _paper_url_index(papers)

    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        normalized = _normalize_link_text(label)
        url = url_index.get(normalized)
        if url is None:
            for key, candidate in url_index.items():
                if key and (key in normalized or normalized in key):
                    url = candidate
                    break
        if url:
            return f"[{label}]({url})"
        return label

    return re.sub(r"\[(?P<label>[^\]]+)\]\(paper_url\)", replace, markdown)


def format_ranked_papers_for_prompt(papers: list[dict]) -> str:
    """Format ranked papers for the agent prompt."""
    lines = []
    for p in papers[:20]:
        pdf_status = "PDF available" if p.get('pdf_exists') else "No PDF"
        primary_url = _paper_primary_url(p)
        arxiv_url = p.get('arxiv_url', '')
        github_url = p.get('github_url', '')
        title = _paper_title(p) or "Untitled paper"

        lines.append(f"Rank {p.get('rank', '?')}: {title}")
        authors = p.get('authors') or 'N/A'
        lines.append(f"  Authors: {authors[:80]}...")
        lines.append(f"  Citations: {p.get('citations', 0)} | {pdf_status}")
        # Include URLs for hyperlink generation
        if primary_url:
            lines.append(f"  URL: {primary_url}")
        if arxiv_url and 'arxiv.org' in arxiv_url:
            lines.append(f"  arXiv: {arxiv_url}")
        elif arxiv_url:
            lines.append(f"  URL: {arxiv_url}")
        if p.get('openreview_url'):
            lines.append(f"  OpenReview: {p.get('openreview_url')}")
        if p.get('forum_url'):
            lines.append(f"  OpenReview: {p.get('forum_url')}")
        if p.get('pdf_url'):
            lines.append(f"  PDF: {p.get('pdf_url')}")
        if github_url:
            lines.append(f"  GitHub: {github_url}")
        lines.append("")
    return "\n".join(lines)


def format_neurips_papers_for_prompt(papers: list[dict]) -> str:
    """Format NeurIPS papers batch for agent prompt."""
    lines = []
    for p in papers[:100]:  # Limit to 100 per batch for context window
        lines.append(f"ID: {p.get('source_id', 'unknown')}")
        lines.append(f"Title: {p.get('source_title', 'Untitled')}")
        # Include URLs if available
        if p.get('arxiv_url'):
            lines.append(f"arXiv: {p.get('arxiv_url')}")
        if p.get('openreview_url'):
            lines.append(f"OpenReview: {p.get('openreview_url')}")
        if p.get('github_url'):
            lines.append(f"GitHub: {p.get('github_url')}")
        lines.append("")
    return "\n".join(lines)


def call_agent(prompt: str, verbose: bool = False) -> dict:
    """Call Claude API with the agent prompt."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed")
        return {}

    client = anthropic.Anthropic()

    if verbose:
        print("    Calling agent...")

    try:
        message = anthropic_messages_create(client, caller=__file__,
            model=default_model_for_provider("presentation.paper_enrichment", "anthropic"),
            max_tokens=4000,
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


def merge_agent_results(
    ranked_result: dict,
    neurips_results: list[dict],
    current_slide_count: int,
    target_slides: int = 60
) -> list[dict]:
    """Merge and deduplicate agent suggestions."""
    new_slides = []
    seen_titles = set()

    slides_needed = target_slides - current_slide_count - 2  # Leave room for papers/closing

    # Priority 1: New slides from ranked papers (pre-curated, high quality)
    for slide in ranked_result.get('new_slides', []):
        title = slide.get('title', '')[:60]
        if title.lower() not in seen_titles and len(new_slides) < slides_needed:
            seen_titles.add(title.lower())
            new_slides.append({
                'title': title,
                'bullets': slide.get('bullets', [])[:5],
                'figure_need': slide.get('figure_need', 'none'),
                'source': slide.get('source', 'Ranked Papers')
            })

    # Priority 2: Gap-filling slides from NeurIPS papers
    for result in neurips_results:
        for proposal in result.get('new_slide_proposals', []):
            title = proposal.get('proposed_slide_title', '')[:60]
            if title.lower() not in seen_titles and len(new_slides) < slides_needed:
                seen_titles.add(title.lower())
                new_slides.append({
                    'title': title,
                    'bullets': proposal.get('bullets', [])[:5],
                    'figure_need': 'none',
                    'source': f"NeurIPS: {proposal.get('title', 'Unknown')[:40]}"
                })

    return new_slides


def generate_enriched_markdown(
    original_md: str,
    new_slides: list[dict],
    paper_mappings: list[dict]
) -> str:
    """Generate the enriched markdown with new slides inserted."""
    lines = original_md.split('\n')
    output_lines = []

    # Find where to insert new slides (before "Top Influential Papers" or at end)
    insert_point = None
    for i, line in enumerate(lines):
        if 'Top Influential Papers' in line or 'Key Takeaways' in line:
            insert_point = i - 2  # Before the --- separator
            break

    if insert_point is None:
        insert_point = len(lines) - 1

    # Copy original content up to insert point
    output_lines = lines[:insert_point]

    # Get current slide count
    last_slide_match = re.findall(r'## Slide (\d+):', original_md)
    next_slide_num = int(last_slide_match[-1]) + 1 if last_slide_match else 1

    # Insert new slides
    for slide in new_slides:
        output_lines.append("")
        output_lines.append(f"## Slide {next_slide_num}: {slide['title']}")
        output_lines.append("")
        for bullet in slide['bullets']:
            output_lines.append(f"- {bullet}")
        output_lines.append("")
        output_lines.append("<!--")
        output_lines.append(f"SOURCE: {slide['source']}")
        if slide.get('figure_need') and slide['figure_need'] != 'none':
            output_lines.append(f"FIGURE_NEED: {slide['figure_need']}")
        output_lines.append("ENRICHMENT_PASS: 2")
        output_lines.append("-->")
        output_lines.append("")
        output_lines.append("---")
        next_slide_num += 1

    # Add remaining original content
    output_lines.extend(lines[insert_point:])

    # Add enrichment metadata
    output_lines.append("")
    output_lines.append(f"<!-- ENRICHED: {datetime.now().isoformat()} -->")
    output_lines.append(f"<!-- NEW_SLIDES_ADDED: {len(new_slides)} -->")

    return '\n'.join(output_lines)


def process_topic(topic_num: str, target_slides: int = 60, verbose: bool = False) -> Path:
    """Process a single topic through enrichment pass."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    if verbose:
        print(f"\nEnriching: {topic_name}")

    # Load inputs
    draft_slides = load_draft_slides(topic_dir)
    ranked_papers = load_ranked_papers(topic_dir)
    neurips_papers = load_neurips_papers(topic_num)

    # Count current slides
    current_slides = len(re.findall(r'## Slide \d+:', draft_slides))
    slides_needed = target_slides - current_slides

    if verbose:
        print(f"  Current slides: {current_slides}")
        print(f"  Target: {target_slides}")
        print(f"  Need: {slides_needed} more slides")
        print(f"  Ranked papers: {len(ranked_papers)}")
        print(f"  NeurIPS papers: {len(neurips_papers)}")

    if slides_needed <= 0:
        if verbose:
            print("  Already at target, skipping enrichment")
        # Just copy to output
        output_path = topic_dir / "2_draft_slides_enriched_with_papers.md"
        enriched_md = replace_unresolved_paper_url_placeholders(
            draft_slides + f"\n\n<!-- ENRICHMENT: No additional slides needed -->\n",
            ranked_papers,
        )
        output_path.write_text(enriched_md)
        return output_path

    # Agent 1: Ranked papers mapper
    if verbose:
        print("  Running Agent 1 (Ranked Papers)...")

    ranked_prompt = RANKED_PAPERS_MAPPER_PROMPT.format(
        draft_slides=draft_slides[:15000],  # Truncate for context
        ranked_papers=format_ranked_papers_for_prompt(ranked_papers)
    )
    ranked_result = call_agent(ranked_prompt, verbose)

    # Agents 2-5: NeurIPS paper discoverers
    neurips_results = []
    if neurips_papers and slides_needed > len(ranked_result.get('new_slides', [])):
        partitions = partition_papers(neurips_papers, 4)
        slide_summaries = extract_slide_summaries(draft_slides)

        for i, batch in enumerate(partitions):
            if not batch:
                continue
            if verbose:
                print(f"  Running Agent {i+2} (NeurIPS batch {i+1}, {len(batch)} papers)...")

            batch_prompt = NEURIPS_PAPER_DISCOVERER_PROMPT.format(
                slide_summaries=slide_summaries,
                papers_batch=format_neurips_papers_for_prompt(batch)
            )
            result = call_agent(batch_prompt, verbose)
            if result:
                neurips_results.append(result)

    # Merge results
    if verbose:
        print("  Merging agent results...")

    new_slides = merge_agent_results(
        ranked_result,
        neurips_results,
        current_slides,
        target_slides
    )

    if verbose:
        print(f"  Generated {len(new_slides)} new slides")

    # Generate enriched markdown
    enriched_md = generate_enriched_markdown(
        draft_slides,
        new_slides,
        ranked_result.get('paper_mappings', [])
    )
    enriched_md = replace_unresolved_paper_url_placeholders(enriched_md, ranked_papers)

    # Save output
    output_path = topic_dir / "2_draft_slides_enriched_with_papers.md"
    output_path.write_text(enriched_md)

    if verbose:
        final_count = len(re.findall(r'## Slide \d+:', enriched_md))
        print(f"  Final slide count: {final_count}")
        print(f"  Saved: {output_path.name}")

    return output_path


class MigrationPathError(SystemExit):
    def __init__(self) -> None:
        super().__init__(2)

    def __str__(self) -> str:
        return MIGRATION_NOTE


def _workspace_relative_path(path: Path, workspace: Path, *, label: str) -> tuple[Path, Path]:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved, relative


def _reject_legacy_presentations_path(relative_path: Path) -> None:
    if "presentations" in relative_path.parts:
        print(MIGRATION_NOTE, file=sys.stderr)
        raise MigrationPathError()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich presentation slides with paper references (Pass 2)"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--draft-slides", type=Path, help="Workspace-relative or absolute draft slides input.")
    parser.add_argument("--matched-papers", type=Path, help="Workspace-relative or absolute canonical matched_papers JSON input.")
    parser.add_argument("--ranked-papers", type=Path, help="Workspace-relative or absolute ranked papers JSON input.")
    parser.add_argument("--presentation-blueprint", type=Path, help="Workspace-relative or absolute approved presentation_blueprint.json input.")
    parser.add_argument("--blueprint-review", type=Path, help="Workspace-relative or absolute approved blueprint_review.json input.")
    parser.add_argument("--output-slides", type=Path, help="Workspace-relative or absolute enriched slides output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate workflow arguments and print planned JSON.")
    parser.add_argument(
        "--topic", type=str,
        help="Topic number (01-20)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all topics"
    )
    parser.add_argument(
        "--slides", type=int, default=60,
        help="Target number of slides (default: 60)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.workspace is not None:
        missing = [
            flag
            for flag, value in (
                ("--presentation-id", args.presentation_id),
                ("--presentation-type", args.presentation_type),
                ("--draft-slides", args.draft_slides),
                ("--matched-papers", args.matched_papers),
                ("--output-slides", args.output_slides),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")
        if not args.dry_run:
            blueprint_missing = [
                flag
                for flag, value in (
                    ("--presentation-blueprint", args.presentation_blueprint),
                    ("--blueprint-review", args.blueprint_review),
                )
                if value is None or value == ""
            ]
            if blueprint_missing:
                parser.error(
                    f"live workspace enrichment requires {', '.join(blueprint_missing)}"
                )

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("draft_slides", "--draft-slides"),
            ("matched_papers", "--matched-papers"),
            ("ranked_papers", "--ranked-papers"),
            ("presentation_blueprint", "--presentation-blueprint"),
            ("blueprint_review", "--blueprint-review"),
            ("output_slides", "--output-slides"),
        ):
            if getattr(args, attr) is None:
                continue
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr in {"output_slides", "presentation_blueprint", "blueprint_review"}:
                _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "draft_slides": str(args.draft_slides),
        "dry_run": True,
        "output_slides": str(args.output_slides),
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "provider_requirements": ENRICHMENT_PROVIDER_REQUIREMENTS,
        "matched_papers": str(args.matched_papers),
        "ranked_papers": str(args.ranked_papers),
        "presentation_blueprint": str(args.presentation_blueprint) if args.presentation_blueprint else None,
        "blueprint_review": str(args.blueprint_review) if args.blueprint_review else None,
        "topic": args.topic,
    }


def _require_blueprint_source_bindings(args: argparse.Namespace) -> bool:
    """Validate optional blueprint source bindings before enrichment writes."""
    if args.presentation_blueprint is None:
        return False
    if args.blueprint_review is None:
        print("--presentation-blueprint requires --blueprint-review", file=sys.stderr)
        raise SystemExit(2)
    result = validate_blueprint_file(args.presentation_blueprint)
    blocking = [
        error
        for error in result.errors
        if error.code in {BLUEPRINT_SOURCE_BINDING_MISSING, BLUEPRINT_SOURCE_UNRESOLVED}
    ]
    if blocking:
        for error in blocking:
            print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(2)
    if result.errors:
        for error in result.errors:
            print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(2)
    try:
        require_approved_blueprint(args.presentation_blueprint, args.blueprint_review)
    except PresentationBlueprintValidationError as exc:
        for error in exc.errors:
            print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(2) from exc
    return True


def process_workspace_enrichment(args: argparse.Namespace) -> Path:
    """Run enrichment from explicit manifest-backed paths without topic discovery."""
    blueprint_bound = _require_blueprint_source_bindings(args)
    draft_slides = args.draft_slides.read_text(encoding="utf-8")
    papers_path = args.matched_papers or args.ranked_papers
    with open(papers_path) as f:
        ranked_data = json.load(f)
    ranked_papers = ranked_data.get("papers", [])

    current_slides = len(re.findall(r'## Slide \d+:', draft_slides))
    if args.verbose:
        print(f"  Current slides: {current_slides}")
        print(f"  Ranked papers: {len(ranked_papers)}")

    if current_slides >= args.slides:
        if blueprint_bound:
            enriched_md = draft_slides + "\n\n<!-- ENRICHMENT: Blueprint source bindings verified -->\n"
        else:
            enriched_md = draft_slides + "\n\n<!-- ENRICHMENT: No additional slides needed -->\n"
    else:
        ranked_prompt = RANKED_PAPERS_MAPPER_PROMPT.format(
            draft_slides=draft_slides[:15000],
            ranked_papers=format_ranked_papers_for_prompt(ranked_papers),
        )
        ranked_result = call_agent(ranked_prompt, args.verbose)
        new_slides = merge_agent_results(
            ranked_result,
            [],
            current_slides,
            args.slides,
        )
        enriched_md = generate_enriched_markdown(
            draft_slides,
            new_slides,
            ranked_result.get('paper_mappings', []),
        )
    enriched_md = replace_unresolved_paper_url_placeholders(enriched_md, ranked_papers)

    args.output_slides.parent.mkdir(parents=True, exist_ok=True)
    args.output_slides.write_text(enriched_md, encoding="utf-8")
    if args.verbose:
        print(f"  Saved: {args.output_slides}")
    return args.output_slides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # D-QUICK-260715-rz1: establish a workspace-rooted provider_receipt_context
    # so standalone invocations don't fall back to Path.cwd() (repo root);
    # --workspace is optional here for the legacy topic-name code path.
    _receipt_workspace = args.workspace or infer_conference_workspace(".")
    if _receipt_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_receipt_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "enrich_slides_with_papers")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        output_path = process_workspace_enrichment(args)
        print(f"Generated: {output_path}")
        return 0

    if args.all:
        print(f"Enriching all topics (target: {args.slides} slides)...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, args.slides, args.verbose)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, args.slides, verbose=True)
        print(f"Generated: {output_path}")

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
