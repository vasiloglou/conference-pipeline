#!/usr/bin/env python3
"""
Hybrid hyperlink enrichment (Pass 4.5) - combines dictionary + agent approaches.

1. First pass: Dictionary-based (fast, catches all known tools)
2. Second pass: Agent-based (finds contextual links the dictionary missed)

This gives the best of both worlds:
- Comprehensive coverage from the dictionary
- Contextual intelligence from the agent

Output: 4.5_draft_slides_with_hyperlinks.md

Usage:
    python scripts/presentations/enrich_hyperlinks_hybrid.py --topic 01
    python scripts/presentations/enrich_hyperlinks_hybrid.py --all
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from datetime import datetime

# Import from both approaches
from enrich_hyperlinks import (
    KNOWN_TOOL_URLS,
    load_url_database as load_dict_database,
    find_linkable_terms,
    apply_hyperlinks,
    process_slide_content,
)
from enrich_hyperlinks_agent import (
    PRIORITY_URLS,
    load_url_database as load_agent_database,
    format_slides_for_prompt,
    format_url_database_for_prompt,
    call_agent,
    IDENTIFY_LINKS_PROMPT,
    get_best_url,
)
from conference_pipeline.presentation.blueprint import (
    PresentationBlueprintValidationError,
    require_approved_blueprint,
    validate_blueprint_file,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
HYPERLINK_PROVIDER_REQUIREMENTS = ["env:ANTHROPIC_API_KEY"]
BLUEPRINT_REFERENCE_LABEL_INVALID = "BLUEPRINT_REFERENCE_LABEL_INVALID"

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


def apply_agent_links(content: str, links_to_add: list[dict]) -> tuple[str, int]:
    """Apply agent-identified hyperlinks to content."""
    links_applied = 0

    # Group links by slide
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

            # Skip if any link already contains this term
            if re.search(r'\[' + re.escape(term) + r'\]', body, re.IGNORECASE):
                continue

            # Use word boundary matching
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, body, re.IGNORECASE):
                actual_match = re.search(pattern, body, re.IGNORECASE)
                if actual_match:
                    actual_term = actual_match.group()
                    body = body[:actual_match.start()] + f'[{actual_term}]({url})' + body[actual_match.end():]
                    links_applied += 1

        return header + body

    result = re.sub(slide_pattern, replace_slide, content, flags=re.DOTALL)
    return result, links_applied


def _markdown_link_count(markdown: str) -> int:
    return len(re.findall(r'\[[^\]]+\]\(https?://[^)]+\)', markdown))


def _iter_deep_research_links(dict_url_db: dict, agent_url_db: dict) -> list[dict[str, str]]:
    """Return deduplicated, presentation-ready links from deep research inputs."""
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str, url: str, *, keywords: list[str] | None = None) -> None:
        label = re.sub(r"\s+", " ", label).strip(" -:")
        url = url.strip()
        if not label or not url.startswith("http"):
            return
        key = (label.lower(), url)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "label": label,
                "url": url,
                "keywords": " ".join(keywords or []),
            }
        )

    for entry in dict_url_db.get("papers", {}).values():
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or entry.get("arxiv_url") or "").strip()
        if not title or not url:
            continue
        label = _short_reference_label(title)
        add(label, url, keywords=[title, label])
        pdf_url = str(entry.get("pdf_url") or "").strip()
        if pdf_url.startswith("http"):
            add(f"{label} PDF", pdf_url, keywords=[title, label, "pdf"])

    for term, entry in agent_url_db.items():
        if not isinstance(entry, dict):
            continue
        url = str(
            entry.get("url")
            or entry.get("arxiv_url")
            or entry.get("github_url")
            or ""
        ).strip()
        add(str(term), url, keywords=[str(term)])

    for label, url in PRIORITY_URLS.items():
        add(label, url, keywords=[label])

    return candidates


def _short_reference_label(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        return ""
    if ":" in title:
        prefix = title.split(":", 1)[0].strip()
        if 2 <= len(prefix) <= 48:
            return prefix
    if " - " in title:
        prefix = title.split(" - ", 1)[0].strip()
        if 2 <= len(prefix) <= 48:
            return prefix
    lowered = title.lower()
    special_labels = (
        ("retrieval head", "Retrieval Heads"),
        ("second opinion", "Second Opinion"),
        ("selective rationale", "Selective Rationale"),
        ("skill-level insights", "Skill Insights"),
        ("episodic memories", "Episodic Memory"),
        ("world model", "MMWorld"),
    )
    for needle, label in special_labels:
        if needle in lowered:
            return label
    words = [word.strip(" ,.;:()[]{}") for word in title.split()]
    useful = [word for word in words if word]
    return " ".join(useful[:4])


def _candidate_score(slide_text: str, candidate: dict[str, str]) -> int:
    haystack = slide_text.lower()
    score = 0
    for token in (candidate["label"], candidate.get("keywords", "")):
        for word in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", token):
            if word.lower() in haystack:
                score += 1
    return score


def ensure_hyperlink_density(
    content: str,
    *,
    target_links_per_slide: float | None,
    dict_url_db: dict,
    agent_url_db: dict,
) -> tuple[str, int]:
    """Densify visible reference links so manifest hyperlink targets are enforceable."""
    if target_links_per_slide is None or target_links_per_slide <= 0:
        return content, 0
    candidates = _iter_deep_research_links(dict_url_db, agent_url_db)
    if not candidates:
        return content, 0

    target_per_slide = max(1, math.ceil(target_links_per_slide))
    added = 0
    slide_pattern = r'(## Slide (\d+): [^\n]+\n)(.*?)(?=## Slide \d+:|$)'

    def process_slide(match: re.Match[str]) -> str:
        nonlocal added
        header = match.group(1)
        body = match.group(3)
        current = _markdown_link_count(body)
        missing = target_per_slide - current
        if missing <= 0:
            return header + body

        existing_urls = set(re.findall(r'\[[^\]]+\]\((https?://[^)]+)\)', body))
        scored = sorted(
            candidates,
            key=lambda item: (-_candidate_score(body, item), item["label"].lower()),
        )
        selected: list[dict[str, str]] = []
        for candidate in scored:
            if len(selected) >= missing:
                break
            if _candidate_score(body, candidate) <= 0:
                continue
            if candidate["url"] in existing_urls:
                continue
            if candidate in selected:
                continue
            selected.append(candidate)
            existing_urls.add(candidate["url"])

        if not selected:
            return header + body

        links = ", ".join(f'[{item["label"]}]({item["url"]})' for item in selected)
        reference_line = f"- References: {links}"
        comment_start = body.find("<!--")
        if comment_start >= 0:
            body = (
                body[:comment_start].rstrip()
                + "\n\n"
                + reference_line
                + "\n\n"
                + body[comment_start:].lstrip()
            )
        else:
            body = body.rstrip() + "\n\n" + reference_line + "\n"
        added += len(selected)
        return header + body

    return re.sub(slide_pattern, process_slide, content, flags=re.DOTALL), added


def _blueprint_source_plan_repair_of(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    value = payload.get("source_plan_repair_of")
    return value if isinstance(value, str) and value.strip() else None


def _require_approved_blueprint_if_present(args: argparse.Namespace) -> bool:
    if args.presentation_blueprint is None:
        return False
    if args.blueprint_review is None:
        print("--presentation-blueprint requires --blueprint-review", file=sys.stderr)
        raise SystemExit(2)
    result = validate_blueprint_file(args.presentation_blueprint)
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


def _approved_labels_from_comment(slide_body: str) -> set[str]:
    match = re.search(r"APPROVED_SOURCE_LABELS:\s*([^\n]+)", slide_body)
    if not match:
        return set()
    return {
        label.strip()
        for label in match.group(1).split(",")
        if label.strip()
    }


def _reject_invalid_blueprint_reference_labels(content: str) -> None:
    slide_pattern = r'(## Slide \d+: [^\n]+\n)(.*?)(?=## Slide \d+:|$)'
    for match in re.finditer(slide_pattern, content, flags=re.DOTALL):
        header = match.group(1).strip()
        body = match.group(2)
        if "APPROVED_SOURCE_LABELS:" not in body:
            continue
        approved_labels = _approved_labels_from_comment(body)
        for reference_line in re.findall(r"(?im)^[-*]\s*References?:\s*(.+)$", body):
            for label in re.findall(r"\[([^\]]+)\]\(https?://[^)]+\)", reference_line):
                if re.fullmatch(r"Ref\s+\d+", label.strip(), flags=re.IGNORECASE):
                    print(
                        f"{BLUEPRINT_REFERENCE_LABEL_INVALID}: {header} uses {label!r}; "
                        "blueprint-bound visible references must use APPROVED_SOURCE_LABELS",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)
                if approved_labels and label.strip() not in approved_labels:
                    print(
                        f"{BLUEPRINT_REFERENCE_LABEL_INVALID}: {header} uses {label!r}; "
                        f"allowed labels are {sorted(approved_labels)} from APPROVED_SOURCE_LABELS",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)


def process_topic(topic_num: str, verbose: bool = False, skip_agent: bool = False) -> Path:
    """Process a single topic with hybrid hyperlink enrichment."""
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
        print(f"\nHybrid hyperlink enrichment: {topic_name}")
        print(f"  Input: {input_path.name}")

    content = input_path.read_text()

    # === PASS 1: Dictionary-based enrichment ===
    if verbose:
        print("  Pass 1: Dictionary-based...")

    # Load dictionary URL database
    dict_url_db = load_dict_database(topic_dir)

    # Count links before
    links_before = content.count('](http')

    # Process with dictionary approach
    slide_pattern = r'(## Slide \d+: [^\n]+\n)(.*?)(?=## Slide \d+:|$)'
    dict_links = 0

    def process_dict_slide(match):
        nonlocal dict_links
        header = match.group(1)
        body = match.group(2)
        enriched_body, links = process_slide_content(body, dict_url_db)
        dict_links += links
        return header + enriched_body

    content = re.sub(slide_pattern, process_dict_slide, content, flags=re.DOTALL)

    if verbose:
        print(f"    Dictionary links added: {dict_links}")

    # === PASS 2: Agent-based enrichment (optional) ===
    agent_links = 0
    if not skip_agent:
        if verbose:
            print("  Pass 2: Agent-based...")

        # Load agent URL database
        agent_url_db = load_agent_database(topic_dir)

        # Format for agent - only slides that don't have many links yet
        slides_for_prompt = format_slides_for_prompt(content)
        url_db_for_prompt = format_url_database_for_prompt(agent_url_db)

        # Call agent
        prompt = IDENTIFY_LINKS_PROMPT.format(
            slides_content=slides_for_prompt,
            url_database=url_db_for_prompt
        )

        result = call_agent(prompt, verbose)
        links_to_add = result.get('links_to_add', [])

        if verbose:
            print(f"    Agent identified: {len(links_to_add)} terms")

        # Apply agent links
        content, agent_links = apply_agent_links(content, links_to_add)

        if verbose:
            print(f"    Agent links applied: {agent_links}")

    # Update metadata
    content = re.sub(
        r'GENERATOR: [^\n]+',
        'GENERATOR: enrich_hyperlinks_hybrid.py (Pass 4.5)',
        content
    )

    # Add timestamp and stats
    total_links = dict_links + agent_links
    content += f"\n<!-- Hyperlinks enriched (hybrid): {datetime.now().isoformat()} -->\n"
    content += f"<!-- Dictionary links: {dict_links}, Agent links: {agent_links}, Total: {total_links} -->\n"

    # Save output
    output_path = topic_dir / "4.5_draft_slides_with_hyperlinks.md"
    output_path.write_text(content)

    if verbose:
        print(f"  Total links added: {total_links}")
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
        description="Hybrid hyperlink enrichment (Pass 4.5)"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--input-slides", type=Path, help="Workspace-relative or absolute sanity-checked slides input.")
    parser.add_argument("--output-slides", type=Path, help="Workspace-relative or absolute hyperlink-enriched slides output.")
    parser.add_argument("--source-attributions", type=Path, help="Workspace-relative or absolute source-attribution JSON input.")
    parser.add_argument("--matched-papers", type=Path, help="Workspace-relative or absolute matched-papers JSON input for deep-research URLs.")
    parser.add_argument("--presentation-blueprint", type=Path, help="Workspace-relative or absolute approved presentation_blueprint.json input.")
    parser.add_argument("--blueprint-review", type=Path, help="Workspace-relative or absolute approved blueprint_review.json input.")
    parser.add_argument("--target-links-per-slide", type=float, help="Minimum visible hyperlink density to enforce per markdown slide.")
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
        "--skip-agent", action="store_true",
        help="Skip agent pass (dictionary only)"
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
                ("--input-slides", args.input_slides),
                ("--output-slides", args.output_slides),
                ("--source-attributions", args.source_attributions),
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
                    f"live workspace hyperlink enrichment requires {', '.join(blueprint_missing)}"
                )

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("input_slides", "--input-slides"),
            ("output_slides", "--output-slides"),
            ("source_attributions", "--source-attributions"),
            ("matched_papers", "--matched-papers"),
            ("presentation_blueprint", "--presentation-blueprint"),
            ("blueprint_review", "--blueprint-review"),
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
        "dry_run": True,
        "input_slides": str(args.input_slides),
        "output_slides": str(args.output_slides),
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "provider_requirements": HYPERLINK_PROVIDER_REQUIREMENTS,
        "matched_papers": str(args.matched_papers) if args.matched_papers else None,
        "presentation_blueprint": str(args.presentation_blueprint) if args.presentation_blueprint else None,
        "blueprint_review": str(args.blueprint_review) if args.blueprint_review else None,
        "source_plan_repair_of": _blueprint_source_plan_repair_of(args.presentation_blueprint),
        "source_attributions": str(args.source_attributions),
        "target_links_per_slide": args.target_links_per_slide,
        "topic": args.topic,
    }


def process_workspace_hyperlinks(args: argparse.Namespace) -> Path:
    """Run hyperlink enrichment from explicit manifest-backed paths."""
    blueprint_bound = _require_approved_blueprint_if_present(args)
    content = args.input_slides.read_text(encoding="utf-8")
    if blueprint_bound:
        _reject_invalid_blueprint_reference_labels(content)
    topic_dir = args.source_attributions.parent

    if args.verbose:
        print(f"  Input: {args.input_slides}")
        print(f"  Source attributions: {args.source_attributions}")

    extra_paper_paths = [args.matched_papers] if args.matched_papers else []
    dict_url_db = load_dict_database(topic_dir, extra_paper_paths=extra_paper_paths)
    slide_pattern = r'(## Slide \d+: [^\n]+\n)(.*?)(?=## Slide \d+:|$)'
    dict_links = 0

    def process_dict_slide(match):
        nonlocal dict_links
        header = match.group(1)
        body = match.group(2)
        enriched_body, links = process_slide_content(body, dict_url_db)
        dict_links += links
        return header + enriched_body

    content = re.sub(slide_pattern, process_dict_slide, content, flags=re.DOTALL)

    agent_url_db = load_agent_database(topic_dir, extra_paper_paths=extra_paper_paths)
    agent_links = 0
    if not args.skip_agent:
        prompt = IDENTIFY_LINKS_PROMPT.format(
            slides_content=format_slides_for_prompt(content),
            url_database=format_url_database_for_prompt(agent_url_db),
        )
        result = call_agent(prompt, args.verbose)
        links_to_add = result.get('links_to_add', [])
        content, agent_links = apply_agent_links(content, links_to_add)

    density_links = 0
    content, density_links = ensure_hyperlink_density(
        content,
        target_links_per_slide=args.target_links_per_slide,
        dict_url_db=dict_url_db,
        agent_url_db=agent_url_db,
    )
    if blueprint_bound:
        _reject_invalid_blueprint_reference_labels(content)

    content = re.sub(
        r'GENERATOR: [^\n]+',
        'GENERATOR: enrich_hyperlinks_hybrid.py (Pass 4.5)',
        content,
    )
    total_links = dict_links + agent_links + density_links
    content += f"\n<!-- Hyperlinks enriched (hybrid): {datetime.now().isoformat()} -->\n"
    content += (
        f"<!-- Dictionary links: {dict_links}, Agent links: {agent_links}, "
        f"Density links: {density_links}, Total: {total_links} -->\n"
    )

    args.output_slides.parent.mkdir(parents=True, exist_ok=True)
    args.output_slides.write_text(content, encoding="utf-8")

    if args.verbose:
        print(f"  Total links added: {total_links}")
        print(f"  Saved: {args.output_slides}")

    return args.output_slides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        output_path = process_workspace_hyperlinks(args)
        print(f"Generated: {output_path}")
        return 0

    if args.all:
        print("Hybrid hyperlink enrichment for all topics...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, args.verbose, args.skip_agent)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, verbose=True, skip_agent=args.skip_agent)
        print(f"Generated: {output_path}")

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
