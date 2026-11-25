#!/usr/bin/env python3
"""
Sanity check presentation slides (Pass 4).

Uses an agent to review slides for:
- Coherent titles that match content
- No truncated or malformed text
- No presentation IDs or metadata artifacts
- No gaps or unusual expressions
- Consistent style and terminology

Input: 3_draft_slides_reflowed.md (or earlier passes)
Output: 4_draft_slides_sanity_check.md

Usage:
    python scripts/presentations/sanity_check_slides.py --topic 01
    python scripts/presentations/sanity_check_slides.py --all
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    infer_conference_workspace,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
SANITY_PROVIDER_REQUIREMENTS = ["python:stdlib"]

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

SANITY_CHECK_PROMPT = '''You are reviewing presentation slides for quality issues.

SLIDES TO REVIEW:
{slides_content}

YOUR TASK:
Review each slide and identify issues. For each issue, provide a fix.

SPECIAL SLIDE TYPES (DO NOT flag these as issues):
- BANNER SLIDES: Slides with "SLIDE_TYPE: banner" are section dividers - they INTENTIONALLY have no bullets
- AGENDA SLIDES: Slides with "SLIDE_TYPE: agenda" list section topics - their bullets are section names

ISSUES TO CHECK:

1. TITLE ISSUES:
   - Title doesn't match bullet content
   - Title is truncated (ends with "...")
   - Title contains metadata artifacts (backticks, "To:", presentation IDs)
   - Title is too vague ("Technical Details", "Overview")
   - Title is too long (>60 chars)

2. BULLET ISSUES:
   - Truncated text (ends mid-sentence)
   - Contains URLs or file paths
   - Contains markdown artifacts (**, *, #, |)
   - Contains numbered list markers embedded ("...1.", "...2.")
   - Too short (<15 chars) or too long (>150 chars)
   - Duplicate content across slides

3. COHERENCE ISSUES:
   - Slide doesn't fit the topic
   - Content is too technical/jargon-heavy without context
   - Missing logical flow between bullets
   - Incomplete thoughts

4. FORMAT ISSUES:
   - Missing bullets (empty slide) - EXCEPT for banner slides
   - Too many bullets (>6)
   - Inconsistent capitalization

OUTPUT FORMAT (JSON):
{{
  "issues_found": [
    {{
      "slide_num": 5,
      "issue_type": "title_mismatch|truncated|artifact|vague|bullet_quality|coherence",
      "description": "Title 'To:' is a metadata artifact",
      "current": "To:",
      "suggested_fix": "Evaluation Infrastructure Trends"
    }}
  ],
  "slides_to_remove": [
    {{
      "slide_num": 15,
      "reason": "Duplicate of slide 3 content"
    }}
  ],
  "overall_quality": "good|needs_fixes|major_issues",
  "summary": "Brief summary of quality assessment"
}}

IMPORTANT: NEVER suggest removing banner slides or agenda slides - they are intentional structural elements.
Be thorough but practical. Only flag genuine issues that would look bad in a presentation.
'''


def parse_slides(content: str) -> list[dict]:
    """Parse markdown into slide objects."""
    slides = []
    slide_pattern = r'## Slide (\d+): ([^\n]+)\n(.*?)(?=## Slide \d+:|$)'

    for match in re.finditer(slide_pattern, content, re.DOTALL):
        num = int(match.group(1))
        title = match.group(2).strip()
        body = match.group(3).strip()

        # Extract bullets
        bullets = []
        for bullet_match in re.finditer(r'^- (.+)$', body, re.MULTILINE):
            bullets.append(bullet_match.group(1).strip())

        # Detect slide type from metadata
        slide_type = 'content'
        if 'SLIDE_TYPE: banner' in body:
            slide_type = 'banner'
        elif 'SLIDE_TYPE: agenda' in body:
            slide_type = 'agenda'

        slides.append({
            'num': num,
            'title': title,
            'bullets': bullets,
            'slide_type': slide_type,
            'raw': match.group(0)
        })

    return slides


def format_slides_for_prompt(slides: list[dict]) -> str:
    """Format slides for agent prompt."""
    lines = []
    for s in slides:
        lines.append(f"=== SLIDE {s['num']} ===")
        if s.get('slide_type') and s['slide_type'] != 'content':
            lines.append(f"SLIDE_TYPE: {s['slide_type']}")
        lines.append(f"TITLE: {s['title']}")
        if s['bullets']:
            lines.append("BULLETS:")
            for b in s['bullets']:
                lines.append(f"  - {b}")
        elif s.get('slide_type') == 'banner':
            lines.append("(Section divider slide - no bullets expected)")
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
        print("    Calling agent...")

    try:
        message = anthropic_messages_create(client, caller=__file__,
            model=default_model_for_provider("presentation.sanity_check", "anthropic"),
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


def apply_fixes(content: str, slides: list[dict], check_result: dict) -> str:
    """Apply fixes from sanity check to the content."""
    # Build maps by both slide number AND title for robust matching
    fixes_by_num = {}
    fixes_by_title = {}
    for issue in check_result.get('issues_found', []):
        slide_num = issue.get('slide_num')
        current_val = issue.get('current', '')

        if slide_num not in fixes_by_num:
            fixes_by_num[slide_num] = []
        fixes_by_num[slide_num].append(issue)

        # Also index by the current value for title fixes
        if 'title' in issue.get('issue_type', '') and current_val:
            fixes_by_title[current_val.lower().strip()] = issue

    # Slides to remove - index by both number and title
    slides_to_remove_nums = set()
    slides_to_remove_titles = set()
    for s in check_result.get('slides_to_remove', []):
        slides_to_remove_nums.add(s['slide_num'])
        # Find the slide title for this number
        for slide in slides:
            if slide['num'] == s['slide_num']:
                slides_to_remove_titles.add(slide['title'].lower().strip())
                break

    # Extract header (before first slide)
    output_lines = []
    header_match = re.match(r'^(.*?)(?=## Slide \d+:)', content, re.DOTALL)
    if header_match:
        output_lines.append(header_match.group(1).rstrip())

    new_slide_num = 1
    for slide in slides:
        original_title = slide['title']
        title_lower = original_title.lower().strip()
        slide_type = slide.get('slide_type', 'content')

        # Never remove banner or agenda slides - they are structural
        is_structural = slide_type in ('banner', 'agenda')

        # Skip removed slides (check both number and title) unless structural
        if not is_structural and (slide['num'] in slides_to_remove_nums or title_lower in slides_to_remove_titles):
            continue

        title = original_title
        bullets = slide['bullets'].copy()  # Copy to avoid modifying original

        # Apply fixes - check by number first, then by title match
        fixes_to_apply = []
        if slide['num'] in fixes_by_num:
            fixes_to_apply.extend(fixes_by_num[slide['num']])

        # Also check if title matches a fix
        if title_lower in fixes_by_title:
            fix = fixes_by_title[title_lower]
            if fix not in fixes_to_apply:
                fixes_to_apply.append(fix)

        for fix in fixes_to_apply:
            issue_type = fix.get('issue_type', '')
            suggested = fix.get('suggested_fix', '')
            current_val = fix.get('current', '')

            if not suggested:
                continue

            # Title fixes
            if 'title' in issue_type or 'artifact' in issue_type or 'vague' in issue_type:
                if current_val and current_val.lower().strip() == title_lower:
                    title = suggested
                elif not current_val:  # No current specified, apply to this slide's title
                    title = suggested

            # Bullet fixes
            elif 'bullet' in issue_type or 'truncated' in issue_type:
                if current_val:
                    for i, b in enumerate(bullets):
                        # Match partial content
                        if current_val in b or b in current_val:
                            bullets[i] = suggested
                            break

        # Write the slide
        output_lines.append("")
        output_lines.append(f"## Slide {new_slide_num}: {title}")
        output_lines.append("")

        # Banner slides get their special comment first
        if slide_type == 'banner':
            output_lines.append("<!-- BANNER SLIDE - Section divider -->")
            output_lines.append("")

        # Write bullets (if any)
        for bullet in bullets:
            output_lines.append(f"- {bullet}")
        if bullets:
            output_lines.append("")

        # Preserve metadata comment (find the one with SLIDE_TYPE, ORIGINAL_NUM, or SOURCE)
        all_comments = re.findall(r'<!--(.*?)-->', slide['raw'], re.DOTALL)
        metadata_comment = None
        for comment in all_comments:
            if any(key in comment for key in ['SLIDE_TYPE:', 'ORIGINAL_NUM:', 'SOURCE:', 'REFLOW_PASS:']):
                metadata_comment = comment.strip()
                break

        if metadata_comment:
            output_lines.append("<!--")
            output_lines.append(metadata_comment)
            output_lines.append("SANITY_CHECK_PASS: 4")
            output_lines.append("-->")

        output_lines.append("")
        output_lines.append("---")

        new_slide_num += 1

    return '\n'.join(output_lines)


def process_topic(topic_num: str, verbose: bool = False) -> Path:
    """Process a single topic through sanity check pass."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    if verbose:
        print(f"\nSanity checking: {topic_name}")

    # Load input (try Pass 3.5 consolidated, fall back to Pass 3, then earlier)
    input_path = topic_dir / "3.5_draft_slides_consolidated.md"
    if not input_path.exists():
        input_path = topic_dir / "3_draft_slides_reflowed.md"
    if not input_path.exists():
        input_path = topic_dir / "2_draft_slides_enriched_with_papers.md"
    if not input_path.exists():
        input_path = topic_dir / "1_draft_slides.md"
    if not input_path.exists():
        raise FileNotFoundError(f"No input slides found in {topic_dir}")

    content = input_path.read_text()
    slides = parse_slides(content)

    if verbose:
        print(f"  Input: {input_path.name}")
        print(f"  Slides to check: {len(slides)}")

    # Format for agent
    slides_content = format_slides_for_prompt(slides)

    # Call agent
    if verbose:
        print("  Running sanity check agent...")

    prompt = SANITY_CHECK_PROMPT.format(slides_content=slides_content)
    check_result = call_agent(prompt, verbose)

    # Report findings
    issues = check_result.get('issues_found', [])
    removals = check_result.get('slides_to_remove', [])
    quality = check_result.get('overall_quality', 'unknown')

    if verbose:
        print(f"  Quality: {quality}")
        print(f"  Issues found: {len(issues)}")
        print(f"  Slides to remove: {len(removals)}")
        if issues:
            for issue in issues[:5]:  # Show first 5
                print(f"    - Slide {issue.get('slide_num')}: {issue.get('description', '')[:50]}")

    # Apply fixes
    if verbose:
        print("  Applying fixes...")

    fixed_content = apply_fixes(content, slides, check_result)

    # Add metadata
    fixed_content += f"\n\n<!-- SANITY_CHECK: {datetime.now().isoformat()} -->"
    fixed_content += f"\n<!-- QUALITY: {quality} -->"
    fixed_content += f"\n<!-- ISSUES_FIXED: {len(issues)} -->"
    fixed_content += f"\n<!-- SLIDES_REMOVED: {len(removals)} -->\n"

    # Save output
    output_path = topic_dir / "4_draft_slides_sanity_check.md"
    output_path.write_text(fixed_content)

    # Save check report
    report_path = topic_dir / "sanity_check_report.json"
    with open(report_path, 'w') as f:
        json.dump(check_result, f, indent=2)

    if verbose:
        final_slides = len(parse_slides(fixed_content))
        print(f"  Output slides: {final_slides}")
        print(f"  Saved: {output_path.name}")
        print(f"  Report: {report_path.name}")

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
        description="Sanity check presentation slides (Pass 4)"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--input-slides", type=Path, help="Workspace-relative or absolute reflowed slides input.")
    parser.add_argument("--output-slides", type=Path, help="Workspace-relative or absolute sanity-checked slides output.")
    parser.add_argument("--report-json", type=Path, help="Workspace-relative or absolute sanity-check report output.")
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
                ("--report-json", args.report_json),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("input_slides", "--input-slides"),
            ("output_slides", "--output-slides"),
            ("report_json", "--report-json"),
        ):
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr in {"output_slides", "report_json"}:
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
        "provider_requirements": SANITY_PROVIDER_REQUIREMENTS,
        "report_json": str(args.report_json),
        "topic": args.topic,
    }


def process_workspace_sanity_check(args: argparse.Namespace) -> Path:
    """Run sanity checks from explicit manifest-backed paths without topic discovery."""
    content = args.input_slides.read_text(encoding="utf-8")
    slides = parse_slides(content)

    if args.verbose:
        print(f"  Input: {args.input_slides}")
        print(f"  Slides to check: {len(slides)}")

    slides_content = format_slides_for_prompt(slides)
    prompt = SANITY_CHECK_PROMPT.format(slides_content=slides_content)
    check_result = call_agent(prompt, args.verbose)

    issues = check_result.get('issues_found', [])
    removals = check_result.get('slides_to_remove', [])
    quality = check_result.get('overall_quality', 'unknown')

    fixed_content = apply_fixes(content, slides, check_result)
    fixed_content += f"\n\n<!-- SANITY_CHECK: {datetime.now().isoformat()} -->"
    fixed_content += f"\n<!-- QUALITY: {quality} -->"
    fixed_content += f"\n<!-- ISSUES_FIXED: {len(issues)} -->"
    fixed_content += f"\n<!-- SLIDES_REMOVED: {len(removals)} -->\n"

    args.output_slides.parent.mkdir(parents=True, exist_ok=True)
    args.output_slides.write_text(fixed_content, encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(check_result, f, indent=2)

    if args.verbose:
        print(f"  Saved: {args.output_slides}")
        print(f"  Report: {args.report_json}")

    return args.output_slides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # D-QUICK-260715-rz1: establish a workspace-rooted provider_receipt_context
    # so standalone invocations don't fall back to Path.cwd() (repo root);
    # --workspace is optional here for the legacy topic-name code path.
    _receipt_workspace = args.workspace or infer_conference_workspace(".")
    if _receipt_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_receipt_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "sanity_check_slides")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        output_path = process_workspace_sanity_check(args)
        print(f"Generated: {output_path}")
        return 0

    if args.all:
        print("Sanity checking all topics...")
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
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
