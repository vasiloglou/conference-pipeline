#!/usr/bin/env python3
"""
Deprecated fail-closed entrypoint for paper-figure extraction.

This script used to extract paper-crop figures for presentation slides. That
path is intentionally disabled for reusable conference production because it
can satisfy planning terms while bypassing the required programmatic
matplotlib/Plotly/svgwrite SVG figure generation contract.

The argument parser remains for backward-compatible inspection and migration,
but `main()` exits fail-closed so this script cannot satisfy
`presentation.svg_figure_generation`.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from conference_pipeline.provider_receipts import (
    openai_chat_completions_create,
)

# Add src to path for importing conference_pipeline
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conference_pipeline.figure_extractor import extract_figures_from_paper

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Topic directories
TOPIC_NAMES = [
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

PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
FIGURE_PROVIDER_REQUIREMENTS = ["python:openai"]
DEPRECATION_ERROR = (
    "DEPRECATED: scripts/presentations/extract_topic_figures.py is disabled for "
    "presentation production. Paper-crop extraction must not satisfy "
    "presentation.svg_figure_generation; implement or run the programmatic "
    "matplotlib/Plotly/svgwrite SVG figure generator instead."
)


def slugify(text: str, max_length: int = 40) -> str:
    """Convert text to a filename-safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower())
    slug = slug.strip('_')
    return slug[:max_length]


def is_recent_paper(paper: dict, min_year: int = 2022) -> bool:
    """Check if a paper was published in min_year or later."""
    published = paper.get('published', '')
    if not published:
        return True
    year_match = re.search(r'(20\d{2})', str(published))
    if year_match:
        return int(year_match.group(1)) >= min_year
    return True


def parse_slide_paper_references(markdown_path: Path) -> dict[int, list[int]]:
    """
    Parse slide markdown to extract paper rank references from SOURCE metadata.

    Looks for patterns like:
        SOURCE: Paper #12: Flow Matching for Generative Modeling
        SOURCE: Papers #9, #10: Diffusion Models

    Returns:
        Dict mapping slide_num -> list of paper rank numbers.
    """
    if not markdown_path.exists():
        return {}

    content = markdown_path.read_text()
    slide_papers: dict[int, list[int]] = {}

    # Parse slide blocks
    slide_pattern = r'## Slide (\d+):[^\n]+'
    slides = list(re.finditer(slide_pattern, content))

    for i, match in enumerate(slides):
        slide_num = int(match.group(1))

        # Get content until next slide or end
        start = match.end()
        end = slides[i + 1].start() if i + 1 < len(slides) else len(content)
        block = content[start:end]

        # Check if this slide needs a figure
        if 'FIGURE_NEED:' not in block:
            continue
        figure_need_match = re.search(r'FIGURE_NEED:\s*(.+)', block)
        if figure_need_match:
            need = figure_need_match.group(1).strip()
            if need == 'none':
                continue

        # Extract paper references from SOURCE field
        source_match = re.search(r'SOURCE:\s*(.+)', block)
        if not source_match:
            continue

        source_text = source_match.group(1).strip()

        # Find all paper rank numbers: "Paper #12", "Papers #9, #10", "Papers #7, 15, 17"
        # First capture explicit #N patterns
        rank_numbers = [int(n) for n in re.findall(r'#(\d+)', source_text)]
        # Also capture bare comma-separated numbers after a #N pattern:
        # "Papers #7, 15, 17: ..." -> captures 15, 17 that lack # prefix
        after_hash = re.search(r'#\d+(.+?)(?::|$)', source_text)
        if after_hash:
            bare_nums = re.findall(r'(?<!\w)(\d{1,3})(?!\w)', after_hash.group(1))
            for n in bare_nums:
                num = int(n)
                if num not in rank_numbers and 1 <= num <= 50:
                    rank_numbers.append(num)
        if rank_numbers:
            slide_papers[slide_num] = rank_numbers

    return slide_papers


def match_slides_to_papers_by_content(
    markdown_path: Path,
    papers: list[dict],
) -> dict[int, list[int]]:
    """
    Match slides to ranked papers using LLM-based semantic matching.

    For slides that don't have explicit Paper #N SOURCE references, sends
    slide content + paper list to GPT-5 mini and asks it to identify the
    most relevant 1-2 papers per slide.

    Falls back to no matches if OpenAI is unavailable.

    Returns:
        Dict mapping slide_num -> list of paper rank numbers matched by content.
    """
    if not markdown_path.exists():
        return {}

    if not HAS_OPENAI:
        print("  Warning: openai package not available, skipping content-based matching")
        return {}

    content = markdown_path.read_text()

    # Collect unmatched slides (have FIGURE_NEED but no Paper #N in SOURCE)
    slide_pattern = r'## Slide (\d+):\s*(.+)'
    slides = list(re.finditer(slide_pattern, content))

    unmatched_slides: list[dict] = []
    for i, match in enumerate(slides):
        slide_num = int(match.group(1))
        slide_title = match.group(2).strip()

        start = match.end()
        end = slides[i + 1].start() if i + 1 < len(slides) else len(content)
        block = content[start:end]

        # Skip slides without FIGURE_NEED
        if 'FIGURE_NEED:' not in block:
            continue
        figure_need_match = re.search(r'FIGURE_NEED:\s*(.+)', block)
        if figure_need_match and figure_need_match.group(1).strip() == 'none':
            continue

        # Skip slides that already have Paper #N references
        source_match = re.search(r'SOURCE:\s*(.+)', block)
        source_text = source_match.group(1).strip() if source_match else ''
        if re.search(r'#\d+', source_text):
            continue

        # Extract bullet content (strip HTML comments and metadata)
        clean_block = re.sub(r'<!--.*?-->', '', block, flags=re.DOTALL)
        # Remove metadata lines
        bullet_lines = []
        for line in clean_block.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith(('SOURCE:', 'FIGURE_NEED:', 'PRESENTATION_ID:')):
                bullet_lines.append(stripped)
        bullet_text = '\n'.join(bullet_lines[:10])  # Cap at 10 lines per slide

        # Extract FIGURE_NEED description
        figure_need = ''
        if figure_need_match:
            figure_need = figure_need_match.group(1).strip()

        unmatched_slides.append({
            'slide_num': slide_num,
            'title': slide_title,
            'content': bullet_text[:500],  # Cap content length
            'figure_need': figure_need,
            'source': source_text,
        })

    if not unmatched_slides:
        return {}

    # Build paper list for the prompt
    paper_list = []
    for p in papers:
        paper_list.append(f"  #{p['rank']}: {p['title']}")
    paper_list_str = '\n'.join(paper_list)

    # Build slide descriptions for the prompt
    slide_descriptions = []
    for s in unmatched_slides:
        desc = f"Slide {s['slide_num']}: \"{s['title']}\"\n"
        if s['figure_need']:
            desc += f"  Figure needed: {s['figure_need']}\n"
        if s['source']:
            desc += f"  Source/theme: {s['source']}\n"
        desc += f"  Content: {s['content'][:300]}"
        slide_descriptions.append(desc)
    slides_str = '\n\n'.join(slide_descriptions)

    # Single batched LLM call
    prompt = f"""You are matching presentation slides to academic papers. For each slide below, identify which paper(s) from the ranked list are most relevant to that slide's content.

RANKED PAPERS:
{paper_list_str}

SLIDES TO MATCH (these slides don't have explicit paper references):
{slides_str}

For each slide, identify the 1-2 most relevant papers by their rank number. Only match papers that are clearly discussed or directly relevant to the slide content. If no paper is a good match, return an empty list for that slide.

Reply with ONLY a JSON object mapping slide numbers to lists of paper rank numbers, like:
{{"10": [3], "36": [4, 5]}}

Return ONLY the JSON, no other text."""

    try:
        client = OpenAI()
        response = openai_chat_completions_create(client, caller=__file__,
            model="gpt-5-mini",  # model-literal-ok: dead code, main() exits fail-closed before this entrypoint is reachable
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
        )
        result_text = response.choices[0].message.content.strip()

        # Parse JSON response - handle markdown code blocks
        if result_text.startswith('```'):
            result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)

        result = json.loads(result_text)

        # Convert to proper types and validate
        matches: dict[int, list[int]] = {}
        valid_ranks = {p['rank'] for p in papers}
        for slide_key, rank_list in result.items():
            slide_num = int(slide_key)
            valid_list = [int(r) for r in rank_list if int(r) in valid_ranks]
            if valid_list:
                matches[slide_num] = valid_list[:2]  # Cap at 2

        return matches

    except Exception as e:
        print(f"  Warning: LLM content matching failed: {e}")
        return {}


def build_papers_by_rank(papers: list[dict]) -> dict[int, dict]:
    """Build a lookup from rank number to paper dict."""
    return {p['rank']: p for p in papers if 'rank' in p}


def find_slides_markdown(topic_dir: Path) -> Path | None:
    """Find the best available slides markdown (latest pass)."""
    for name in [
        "4.5_draft_slides_with_hyperlinks.md",
        "4_draft_slides_sanity_check.md",
        "3_draft_slides_reflowed.md",
        "2_draft_slides_enriched_with_papers.md",
        "1_draft_slides.md",
    ]:
        p = topic_dir / name
        if p.exists():
            return p
    return None


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


def extract_from_explicit_paths(
    *,
    topic_name: str,
    slides_md: Path | None,
    ranked_path: Path,
    figure_output_root: Path,
    inventory_path: Path,
    max_papers_per_slide: int = 2,
    generate_captions: bool = True,
    filter_old_papers: bool = True,
    verbose: bool = True,
) -> dict:
    """Extract figures using explicit manifest-backed input and output paths."""
    if not ranked_path.exists():
        if verbose:
            print(f"No matched_papers.json or ranked_papers.json for {topic_name}: {ranked_path}")
        return {"error": "No matched_papers.json or ranked_papers.json found"}

    # Load ranked papers
    with open(ranked_path) as f:
        data = json.load(f)

    papers = data.get('papers', [])
    papers_by_rank = build_papers_by_rank(papers)

    # Find the slide markdown and parse paper references
    md_path = slides_md if slides_md and slides_md.exists() else None
    slide_paper_refs = parse_slide_paper_references(md_path) if md_path else {}

    # Phase 0: Content-based paper matching for slides without explicit Paper #N refs
    content_matched = match_slides_to_papers_by_content(md_path, papers) if md_path else {}
    content_match_count = 0
    for slide_num, ranks in content_matched.items():
        if slide_num not in slide_paper_refs:
            slide_paper_refs[slide_num] = ranks
            content_match_count += 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"Topic: {topic_name}")
        print(f"Papers available: {len(papers)}")
        print(f"Slides with SOURCE refs: {len(slide_paper_refs) - content_match_count}")
        print(f"Slides matched by LLM: {content_match_count}")
        if slide_paper_refs:
            for sn, ranks in sorted(slide_paper_refs.items()):
                titles = [papers_by_rank[r]['title'][:40] + '...' for r in ranks if r in papers_by_rank]
                label = " (LLM)" if sn in content_matched else ""
                print(f"    Slide {sn:2d} -> Paper(s) #{', #'.join(str(r) for r in ranks)}: {'; '.join(titles)}{label}")
        print(f"Captions: {'enabled (GPT-5.4 mini)' if generate_captions else 'disabled'}")
        print(f"{'='*60}")

    all_figures = []
    # Cache: pdf_path -> list of extracted figure dicts (avoids re-extracting same PDF)
    pdf_extraction_cache: dict[str, list[dict]] = {}
    processed_count = 0
    skipped_count = 0

    def _extract_paper(paper: dict, output_dir: Path) -> list[dict]:
        """Extract figures from a paper, using cache to avoid duplicate extraction."""
        pdf_path = paper.get('pdf_path')
        if not pdf_path or not Path(pdf_path).exists():
            return []

        if pdf_path in pdf_extraction_cache:
            return pdf_extraction_cache[pdf_path]

        try:
            results = extract_figures_from_paper(
                pdf_path=pdf_path,
                output_dir=str(output_dir),
                model="gpt-5.4-mini",  # model-literal-ok: dead code, main() exits fail-closed before this entrypoint is reachable
                stop_at_references=True,
                generate_captions=generate_captions,
                verbose=False
            )
            for fig in results:
                fig['source_paper'] = paper['title']
                fig['paper_rank'] = paper['rank']
                fig['paper_slug'] = slugify(paper['title'])
            pdf_extraction_cache[pdf_path] = results
            return results
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            pdf_extraction_cache[pdf_path] = []
            return []

    # Phase 1: Slide-aware extraction (extract from papers referenced by specific slides)
    for slide_num, rank_list in sorted(slide_paper_refs.items()):
        output_dir = figure_output_root / f"slide_{slide_num:02d}"

        for rank in rank_list[:max_papers_per_slide]:
            paper = papers_by_rank.get(rank)
            if not paper:
                if verbose:
                    print(f"  Slide {slide_num}: Paper #{rank} not found in rankings")
                continue

            pdf_path = paper.get('pdf_path')
            if not pdf_path or not Path(pdf_path).exists():
                if verbose:
                    print(f"  Slide {slide_num}: Paper #{rank} has no PDF")
                skipped_count += 1
                continue

            was_cached = pdf_path in pdf_extraction_cache

            if verbose:
                label = "cached" if was_cached else "extracting"
                print(f"\n  Slide {slide_num} <- Paper #{rank} ({label}): {paper['title'][:55]}...")

            results = _extract_paper(paper, output_dir)

            # Tag each figure with the slide it was extracted for
            slide_figures = []
            for fig in results:
                fig_copy = dict(fig)
                fig_copy['source_slide'] = f"slide_{slide_num:02d}"
                slide_figures.append(fig_copy)

            all_figures.extend(slide_figures)
            if not was_cached:
                processed_count += 1

            if verbose:
                print(f"    Figures: {len(results)}{' (from cache)' if was_cached else ''}")

    # Phase 2: General inventory from top papers not yet covered
    # This provides fallback figures for slides with thematic (non-paper) sources
    covered_ranks = set()
    for ranks in slide_paper_refs.values():
        covered_ranks.update(ranks)

    uncovered_papers = [p for p in papers if p['rank'] not in covered_ranks]

    # Filter old papers
    if filter_old_papers:
        uncovered_papers = [p for p in uncovered_papers if is_recent_paper(p, min_year=2022)]

    for paper in uncovered_papers[:5]:  # Up to 5 additional papers for general inventory
        pdf_path = paper.get('pdf_path')
        if not pdf_path or not Path(pdf_path).exists():
            continue

        # Skip if we already extracted this PDF in phase 1
        if pdf_path in pdf_extraction_cache:
            continue

        slug = slugify(paper['title'])
        output_dir = figure_output_root / slug

        if verbose:
            print(f"\n  General: Paper #{paper['rank']}: {paper['title'][:55]}...")

        results = _extract_paper(paper, output_dir)
        all_figures.extend(results)
        if results:
            processed_count += 1
        else:
            skipped_count += 1

        if verbose:
            print(f"    Extracted: {len(results)} figures")

    # Deduplicate figures: same image_path can appear for multiple slides.
    # Keep unique by (image_path, source_slide) so the inventory tracks which
    # slides each figure is relevant to without repeating extraction metadata.
    seen_keys: set[str] = set()
    deduped_figures = []
    for fig in all_figures:
        img_path = fig.get('image_path', fig.get('path', ''))
        src_slide = fig.get('source_slide', '')
        key = f"{img_path}|{src_slide}"
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_figures.append(fig)

    # Save asset inventory
    inventory = {
        "topic": topic_name,
        "papers_processed": processed_count,
        "papers_skipped": skipped_count,
        "total_figures": len(deduped_figures),
        "slide_paper_mapping": {str(k): v for k, v in slide_paper_refs.items()},
        "figures": deduped_figures,
        "talk_slides": []
    }

    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inventory_path, 'w') as f:
        json.dump(inventory, f, indent=2)

    if verbose:
        print(f"\n  Summary:")
        print(f"    Papers processed: {processed_count}")
        print(f"    Papers skipped: {skipped_count}")
        print(f"    Total figures: {len(all_figures)}")
        print(f"    Saved: {inventory_path}")

    return {
        "topic": topic_name,
        "papers_processed": processed_count,
        "papers_skipped": skipped_count,
        "total_figures": len(all_figures),
    }


def extract_for_topic(
    topic_name: str,
    max_papers_per_slide: int = 2,
    generate_captions: bool = True,
    filter_old_papers: bool = True,
    verbose: bool = True
) -> dict:
    """
    Extract figures from papers for a topic, guided by slide-to-paper mapping.

    For each slide that needs a figure, looks at the SOURCE metadata to find
    which ranked papers it references, then extracts figures from those specific
    papers into slide_XX/ directories.

    Also extracts from top-ranked papers that aren't referenced by any specific
    slide (general inventory).
    """
    topic_dir = PRESENTATIONS_DIR / topic_name
    return extract_from_explicit_paths(
        topic_name=topic_name,
        slides_md=find_slides_markdown(topic_dir),
        ranked_path=(
            topic_dir / "matched_papers.json"
            if (topic_dir / "matched_papers.json").exists()
            else topic_dir / "ranked_papers.json"
        ),
        figure_output_root=topic_dir / "assets" / "figures",
        inventory_path=topic_dir / "asset_inventory.json",
        max_papers_per_slide=max_papers_per_slide,
        generate_captions=generate_captions,
        filter_old_papers=filter_old_papers,
        verbose=verbose,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract figures from papers for presentation topics (slide-aware)"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Conference workspace root for manifest-backed runs.",
    )
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--slides-md", type=Path, help="Workspace-relative or absolute consolidated slides input.")
    parser.add_argument("--matched-papers", type=Path, help="Workspace-relative or absolute canonical matched_papers JSON input.")
    parser.add_argument("--ranked-papers", type=Path, help="Workspace-relative or absolute ranked papers JSON input.")
    parser.add_argument("--output-dir", type=Path, help="Workspace-relative or absolute figure output directory.")
    parser.add_argument("--asset-inventory", type=Path, help="Workspace-relative or absolute asset inventory output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate workflow arguments and print planned JSON.")
    parser.add_argument(
        "--topic", type=str,
        help="Topic number (01-20) or full name (e.g., 01_tools_and_open_source)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all topics"
    )
    parser.add_argument(
        "--max-papers", type=int, default=2,
        help="Maximum papers to extract per slide (default: 2)"
    )
    parser.add_argument(
        "--no-captions", action="store_true",
        help="Skip caption generation (faster, for testing)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--include-old-papers", action="store_true",
        help="Include papers from before 2022 (by default, only 2022+ papers are processed)"
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
                ("--slides-md", args.slides_md),
                ("--matched-papers", args.matched_papers),
                ("--output-dir", args.output_dir),
                ("--asset-inventory", args.asset_inventory),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("slides_md", "--slides-md"),
            ("matched_papers", "--matched-papers"),
            ("ranked_papers", "--ranked-papers"),
            ("output_dir", "--output-dir"),
            ("asset_inventory", "--asset-inventory"),
        ):
            if getattr(args, attr) is None:
                continue
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr in {"output_dir", "asset_inventory"}:
                _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "asset_inventory": str(args.asset_inventory),
        "dry_run": True,
        "output_dir": str(args.output_dir),
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "provider_requirements": FIGURE_PROVIDER_REQUIREMENTS,
        "matched_papers": str(args.matched_papers),
        "ranked_papers": str(args.ranked_papers),
        "slides_md": str(args.slides_md),
        "topic": args.topic,
    }


def process_workspace_figures(args: argparse.Namespace) -> dict:
    """Extract figures from explicitly selected workspace artifacts."""
    return extract_from_explicit_paths(
        topic_name=args.presentation_id,
        slides_md=args.slides_md,
        ranked_path=args.matched_papers or args.ranked_papers,
        figure_output_root=args.output_dir,
        inventory_path=args.asset_inventory,
        max_papers_per_slide=args.max_papers,
        generate_captions=not args.no_captions,
        filter_old_papers=not args.include_old_papers,
        verbose=not args.quiet,
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    print(DEPRECATION_ERROR, file=sys.stderr)
    return 2


def _legacy_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        result = process_workspace_figures(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if "error" not in result else 1

    if args.all:
        print("Processing all topics (slide-aware extraction)...")
        if not args.include_old_papers:
            print("Note: Only processing papers from 2022 or later (use --include-old-papers to include older)")
        results = []
        for topic_name in TOPIC_NAMES:
            try:
                result = extract_for_topic(
                    topic_name=topic_name,
                    max_papers_per_slide=args.max_papers,
                    generate_captions=not args.no_captions,
                    filter_old_papers=not args.include_old_papers,
                    verbose=not args.quiet
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {topic_name}: {e}")
                results.append({"topic": topic_name, "error": str(e)})

        print("\n" + "=" * 60)
        print("Overall Summary:")
        total_papers = sum(r.get('papers_processed', 0) for r in results)
        total_figures = sum(r.get('total_figures', 0) for r in results)
        print(f"  Topics processed: {len(results)}")
        print(f"  Total papers: {total_papers}")
        print(f"  Total figures: {total_figures}")

    elif args.topic:
        topic_input = args.topic.zfill(2) if args.topic.isdigit() else args.topic

        topic_name = None
        for name in TOPIC_NAMES:
            if name.startswith(topic_input) or topic_input in name:
                topic_name = name
                break

        if not topic_name:
            print(f"Unknown topic: {args.topic}")
            print(f"Available topics: {', '.join(TOPIC_NAMES)}")
            sys.exit(1)

        extract_for_topic(
            topic_name=topic_name,
            max_papers_per_slide=args.max_papers,
            generate_captions=not args.no_captions,
            filter_old_papers=not args.include_old_papers,
            verbose=not args.quiet
        )

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
