#!/usr/bin/env python3
"""
Consolidate slides to remove cross-section repetition (Pass 3.5).

Reads 3_draft_slides_reflowed.md and performs a single-pass analysis of
the FULL deck to:
- Detect concept-level repetition across sections
- Regroup slides so related content is co-located
- Merge thin slides (1-2 bullets) on the same concept
- Ensure progressive narrative (introduce once, build upon)

Unlike the reflow pass (which uses 3 specialized agents on truncated content),
this pass sends the COMPLETE slide deck to one LLM call for holistic awareness.

Output: 3.5_draft_slides_consolidated.md

Usage:
    python scripts/presentations/consolidate_slides.py --topic 01
    python scripts/presentations/consolidate_slides.py --all
    python scripts/presentations/consolidate_slides.py --topic 02 -v
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conference_pipeline.model_governance import default_model, validate_selection
from conference_pipeline.conference_identity import conference_name

PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
CONSOLIDATION_PROVIDER_REQUIREMENTS = ["env:GEMINI_API_KEY|GOOGLE_API_KEY"]

MODEL_TASK_ID = "presentation.deck_consolidation"
DEFAULT_MODEL = default_model(MODEL_TASK_ID)

# Topic names (shared across presentation scripts)
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


# ---------------------------------------------------------------------------
# Slide parsing (same pattern as reflow_slides.py / generate_presenter_script.py)
# ---------------------------------------------------------------------------


def parse_slides(content: str) -> list[dict]:
    """Parse slide markdown into a list of raw slide blocks.

    Each block contains the full raw markdown text (header + body) so that
    the LLM can return it verbatim with modifications.
    """
    # Split on slide headers, keeping the header as part of the block
    pattern = r"(?=^## Slide \d+:)"
    blocks = re.split(pattern, content, flags=re.MULTILINE)

    slides: list[dict] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        header_match = re.match(r"^## Slide (\d+): (.+)", block)
        if not header_match:
            continue

        slide_num = int(header_match.group(1))
        title = header_match.group(2).strip()

        # Extract section from metadata
        section = ""
        section_match = re.search(r"SECTION:\s*(.+)", block)
        if section_match:
            section = section_match.group(1).strip()

        # Extract slide type
        slide_type = ""
        type_match = re.search(r"SLIDE_TYPE:\s*(.+)", block)
        if type_match:
            slide_type = type_match.group(1).strip()

        # Extract bullets (lines starting with -)
        bullets = re.findall(r"^- (.+)", block, re.MULTILINE)

        slides.append(
            {
                "num": slide_num,
                "title": title,
                "section": section,
                "slide_type": slide_type,
                "bullets": bullets,
                "raw": block,
            }
        )

    return slides


def extract_header(content: str) -> str:
    """Extract everything before the first slide."""
    match = re.match(r"^(.*?)(?=## Slide \d+:)", content, re.DOTALL)
    return match.group(1) if match else ""


def format_slide_summary(slides: list[dict]) -> str:
    """Format slides as a compact summary for the analysis prompt."""
    lines = []
    for s in slides:
        section_tag = f"  [Section: {s['section']}]" if s["section"] else ""
        type_tag = f"  [Type: {s['slide_type']}]" if s["slide_type"] else ""
        lines.append(f"Slide {s['num']}: {s['title']}{section_tag}{type_tag}")
        for b in s["bullets"][:5]:
            lines.append(f"  - {b[:120]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def call_gemini(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    verbose: bool = False,
) -> str:
    """Call Gemini API with retry logic. Returns raw text response."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit(
            "Error: google-genai package not installed. Run: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    validate_selection(MODEL_TASK_ID, provider="gemini", model=model)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = gemini_generate_content(client, caller=__file__,
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                ),
            )
            text = getattr(response, "text", None)
            if text:
                return text
            if verbose:
                print("    Gemini returned no text; using empty response")
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (2**attempt)
                if verbose:
                    print(
                        f"    Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}"
                    )
                time.sleep(wait_time)
            else:
                raise


# ---------------------------------------------------------------------------
# Consolidation logic
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are a presentation editor consolidating a {conference_label} conference slide deck.

Below is the FULL slide deck. Your job is to identify cross-section concept repetition and
propose a consolidation plan.

SLIDE DECK:
{slide_summary}

TASK: Analyze the deck and return a JSON object with:

1. "concept_clusters": A list of concept groups where the same idea appears on multiple slides.
   Each cluster has:
   - "concept": Short name of the repeated concept
   - "slide_nums": List of slide numbers that discuss this concept
   - "primary_slide": The slide number that covers this concept best (keep this one)
   - "action": One of "merge_into_primary" | "move_together" | "keep_separate"
   - "reason": Why this action

2. "regroup_moves": A list of slides that should move to a different section for better coherence.
   Each move has:
   - "slide_num": The slide to move
   - "from_section": Current section name
   - "to_section": Target section name
   - "reason": Why this move improves coherence

3. "thin_merges": A list of slide pairs where a thin slide (1-2 bullets) should merge into
   a neighbor on the same topic.
   Each merge has:
   - "absorb_slide": The thin slide number (will be removed)
   - "into_slide": The target slide number (will gain the bullets)
   - "reason": Why

4. "delete_slides": Slide numbers to remove entirely (fully redundant, add nothing new).

RULES:
- Be AGGRESSIVE about detecting concept repetition — flag any case where the same paper,
  framework, benchmark, or theme is discussed substantively on 2+ slides in DIFFERENT sections
- But be CONSERVATIVE about deleting — prefer merging over deleting
- Do NOT touch banner slides, agenda slides, or appendix slides (type: banner, agenda, appendix)
- Do NOT break hyperlinks or source references — those must be preserved in any merge
- Maximum 30% of content slides can be affected (merged, moved, or deleted)
- Return ONLY valid JSON, no markdown fences, no explanation outside the JSON

Return the JSON object now."""

CONSOLIDATION_PROMPT = """You are a presentation editor. Below is a slide deck and a consolidation plan.
Apply the plan to produce a consolidated slide deck.

ORIGINAL SLIDE DECK (full markdown):
{full_deck}

CONSOLIDATION PLAN:
{plan_json}

TASK: Apply the consolidation plan and output the COMPLETE consolidated slide deck in markdown.

RULES:
1. For "merge_into_primary" clusters: Keep the primary slide, merge unique bullets from
   other slides into it (max 6 bullets per slide). Remove the absorbed slides entirely.
   Preserve ALL hyperlinks from both slides.

2. For "move_together" clusters: Move the slides so they are adjacent in the deck.
   Update their SECTION metadata to match their new location.

3. For "regroup_moves": Move the slide to the target section. Update its SECTION metadata.
   Place it at a logical position within the target section (near related slides).

4. For "thin_merges": Move bullets from the thin slide into the target slide. Remove the
   thin slide. Preserve all hyperlinks and source refs.

5. For "delete_slides": Remove these slides entirely.

6. After all changes, RENUMBER all slides sequentially starting from 0.
   Update every "## Slide N:" header to reflect the new number.

7. Preserve ALL HTML-comment metadata blocks (<!-- ... -->) exactly as-is, except:
   - Update SECTION values for moved slides
   - Add CONSOLIDATION_NOTE to any slide that absorbed content from another

8. Preserve the document header (everything before the first slide) exactly as-is.

9. Add a CONSOLIDATION_NOTE in the metadata of affected slides:
   <!-- CONSOLIDATION_NOTE: Merged content from original slides X, Y -->

10. Do NOT invent new content. Only rearrange, merge, or remove existing content.

11. Do NOT touch banner slides, agenda slides, or appendix slides.

12. Output the COMPLETE markdown document — every slide, not just the changed ones.
    Start with the header, then all slides in order, exactly as the original format."""


def analyze_deck(
    slides: list[dict],
    *,
    conference_label: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> dict:
    """Send the full deck to the LLM for concept analysis."""
    summary = format_slide_summary(slides)
    prompt = ANALYSIS_PROMPT.format(slide_summary=summary, conference_label=conference_label)

    if verbose:
        print("  Running concept analysis...")

    response = call_gemini(prompt, model=model, max_tokens=4096, verbose=verbose)

    # Extract JSON
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            if verbose:
                print(f"    JSON parse error: {e}")
            return {}
    return {}


def apply_consolidation(
    full_deck: str,
    plan: dict,
    *,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> str:
    """Send the full deck + plan to the LLM to produce consolidated output."""
    plan_json = json.dumps(plan, indent=2)
    prompt = CONSOLIDATION_PROMPT.format(full_deck=full_deck, plan_json=plan_json)

    if verbose:
        clusters = len(plan.get("concept_clusters", []))
        moves = len(plan.get("regroup_moves", []))
        thin = len(plan.get("thin_merges", []))
        deletes = len(plan.get("delete_slides", []))
        print(f"  Applying consolidation: {clusters} clusters, {moves} moves, {thin} thin merges, {deletes} deletes")

    # Use high max_tokens since we need the full deck back
    response = call_gemini(prompt, model=model, max_tokens=65536, verbose=verbose)
    return response


def _is_trivial_plan(plan: dict) -> bool:
    """Check if the plan has nothing to do."""
    clusters = [
        c
        for c in plan.get("concept_clusters", [])
        if c.get("action") != "keep_separate"
    ]
    moves = plan.get("regroup_moves", [])
    thin = plan.get("thin_merges", [])
    deletes = plan.get("delete_slides", [])
    return not clusters and not moves and not thin and not deletes


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def process_topic(topic_num: str, verbose: bool = False, model: str = DEFAULT_MODEL) -> Path:
    """Consolidate slides for a single topic."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    # Find input (prefer 3, fall back to 2, then 1)
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
        print(f"\nConsolidating: {topic_name}")
        print(f"  Input: {input_path.name} ({len(slides)} slides)")

    # Phase 1: Analyze the deck for concept repetition
    # This legacy code path is hardwired to PRESENTATIONS_DIR (conferences/
    # neurips2025/presentations) above -- not a hardcoded-conference bug,
    # since it never runs against any other conference's workspace.
    plan = analyze_deck(slides, conference_label="NeurIPS 2025", model=model, verbose=verbose)

    if not plan:
        if verbose:
            print("  Analysis returned empty — skipping consolidation")
        # Copy input to output unchanged
        output_path = topic_dir / "3.5_draft_slides_consolidated.md"
        output_path.write_text(content)
        return output_path

    if verbose:
        clusters = plan.get("concept_clusters", [])
        actionable = [c for c in clusters if c.get("action") != "keep_separate"]
        print(f"  Concept clusters: {len(clusters)} ({len(actionable)} actionable)")
        for c in clusters:
            action = c.get("action", "?")
            concept = c.get("concept", "?")
            nums = c.get("slide_nums", [])
            print(f"    {action}: {concept} (slides {nums})")
        moves = plan.get("regroup_moves", [])
        if moves:
            print(f"  Regroup moves: {len(moves)}")
            for m in moves:
                print(f"    Slide {m.get('slide_num')}: {m.get('from_section')} -> {m.get('to_section')}")
        thin = plan.get("thin_merges", [])
        if thin:
            print(f"  Thin merges: {len(thin)}")
        deletes = plan.get("delete_slides", [])
        if deletes:
            print(f"  Deletes: {deletes}")

    # Check if there's anything to do
    if _is_trivial_plan(plan):
        if verbose:
            print("  No actionable changes — copying input unchanged")
        output_path = topic_dir / "3.5_draft_slides_consolidated.md"
        output_path.write_text(content)
        return output_path

    # Phase 2: Apply the consolidation plan
    # Send the full deck (not truncated) so the LLM can return a complete output
    consolidated = apply_consolidation(content, plan, model=model, verbose=verbose)

    if not consolidated or len(consolidated) < 500:
        if verbose:
            print("  Consolidation returned too little content — keeping original")
        output_path = topic_dir / "3.5_draft_slides_consolidated.md"
        output_path.write_text(content)
        return output_path

    # Strip markdown fences if the LLM wrapped the output
    consolidated = re.sub(r"^```(?:markdown)?\s*\n", "", consolidated)
    consolidated = re.sub(r"\n```\s*$", "", consolidated)

    # Validate: count slides in output
    output_slides = parse_slides(consolidated)
    removed = len(slides) - len(output_slides)

    if verbose:
        print(f"  Output: {len(output_slides)} slides (removed {removed})")

    # Safety check: don't allow more than 30% removal
    max_removal = int(len(slides) * 0.30)
    if removed > max_removal:
        if verbose:
            print(
                f"  WARNING: Removed {removed} slides (>{max_removal} limit) — keeping original"
            )
        output_path = topic_dir / "3.5_draft_slides_consolidated.md"
        output_path.write_text(content)
        return output_path

    # Update header metadata
    now = datetime.now().isoformat()
    if "GENERATOR:" in consolidated:
        consolidated = re.sub(
            r"GENERATOR:.*",
            f"GENERATOR: consolidate_slides.py (Pass 3.5)",
            consolidated,
            count=1,
        )
    if "PASS:" in consolidated:
        consolidated = re.sub(r"PASS:.*", f"PASS: 3.5", consolidated, count=1)

    # Write output
    output_path = topic_dir / "3.5_draft_slides_consolidated.md"
    output_path.write_text(consolidated)

    # Save the plan for reference
    plan_path = topic_dir / "consolidation_plan.json"
    plan["_metadata"] = {
        "generated": now,
        "input": input_path.name,
        "input_slides": len(slides),
        "output_slides": len(output_slides),
        "removed": removed,
    }
    plan_path.write_text(json.dumps(plan, indent=2))

    if verbose:
        print(f"  Saved: {output_path.name}")
        print(f"  Plan: {plan_path.name}")

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
        description="Consolidate slides to remove cross-section repetition (Pass 3.5)"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--input-slides", type=Path, help="Workspace-relative or absolute hyperlink-enriched slides input.")
    parser.add_argument("--output-slides", type=Path, help="Workspace-relative or absolute consolidated slides output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate workflow arguments and print planned JSON.")
    parser.add_argument("--topic", type=str, help="Topic number (01-20)")
    parser.add_argument("--all", action="store_true", help="Process all topics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
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
        ):
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr == "output_slides":
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
        "provider_requirements": CONSOLIDATION_PROVIDER_REQUIREMENTS,
        "topic": args.topic,
    }


def process_workspace_consolidation(args: argparse.Namespace) -> Path:
    """Consolidate explicitly selected workspace slides without topic discovery."""
    content = args.input_slides.read_text(encoding="utf-8")
    slides = parse_slides(content)
    # args.workspace is guaranteed non-None here (see parse_args's
    # --workspace-requires validation), so this reads the real, current
    # conference's own label -- never a different conference's hardcoded name.
    conference_label = conference_name(args.workspace, default="the conference")

    if args.verbose:
        print(f"  Input: {args.input_slides} ({len(slides)} slides)")

    plan = analyze_deck(
        slides, conference_label=conference_label, model=args.model, verbose=args.verbose
    )
    if not plan or _is_trivial_plan(plan):
        if args.verbose:
            print("  No actionable changes - copying input unchanged")
        args.output_slides.parent.mkdir(parents=True, exist_ok=True)
        args.output_slides.write_text(content, encoding="utf-8")
        return args.output_slides

    consolidated = apply_consolidation(content, plan, model=args.model, verbose=args.verbose)
    if not consolidated or len(consolidated) < 500:
        if args.verbose:
            print("  Consolidation returned too little content - keeping original")
        consolidated = content
    else:
        consolidated = re.sub(r"^```(?:markdown)?\s*\n", "", consolidated)
        consolidated = re.sub(r"\n```\s*$", "", consolidated)
        output_slides = parse_slides(consolidated)
        removed = len(slides) - len(output_slides)
        max_removal = int(len(slides) * 0.30)
        if removed > max_removal:
            if args.verbose:
                print(
                    f"  WARNING: Removed {removed} slides (>{max_removal} limit) - keeping original"
                )
            consolidated = content

    now = datetime.now().isoformat()
    if "GENERATOR:" in consolidated:
        consolidated = re.sub(
            r"GENERATOR:.*",
            "GENERATOR: consolidate_slides.py (Pass 3.5)",
            consolidated,
            count=1,
        )
    if "PASS:" in consolidated:
        consolidated = re.sub(r"PASS:.*", "PASS: 3.5", consolidated, count=1)

    args.output_slides.parent.mkdir(parents=True, exist_ok=True)
    args.output_slides.write_text(consolidated, encoding="utf-8")

    plan_path = args.output_slides.parent / "consolidation_plan.json"
    output_count = len(parse_slides(consolidated))
    plan["_metadata"] = {
        "generated": now,
        "input": args.input_slides.name,
        "input_slides": len(slides),
        "output_slides": output_count,
        "removed": len(slides) - output_count,
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"  Saved: {args.output_slides}")
        print(f"  Plan: {plan_path}")

    return args.output_slides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # D-QUICK-260715-rz1: establish a workspace-rooted provider_receipt_context
    # so standalone invocations don't fall back to Path.cwd() (repo root);
    # --workspace is optional here for the legacy topic-name code path.
    _receipt_workspace = args.workspace or infer_conference_workspace(".")
    if _receipt_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_receipt_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "consolidate_slides")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        output_path = process_workspace_consolidation(args)
        print(f"Generated: {output_path}")
        return 0

    if args.all:
        print(f"Consolidating all topics using {args.model}...")
        for topic_num in TOPIC_NAMES:
            try:
                output = process_topic(topic_num, args.verbose, args.model)
                print(f"  [OK] {TOPIC_NAMES[topic_num]}")
            except FileNotFoundError as e:
                print(f"  [SKIP] {TOPIC_NAMES[topic_num]}: {e}")
            except Exception as e:
                print(f"  [FAIL] {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)
        output_path = process_topic(topic_num, verbose=True, model=args.model)
        print(f"Generated: {output_path}")

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
