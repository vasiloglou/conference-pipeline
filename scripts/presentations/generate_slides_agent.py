#!/usr/bin/env python3
"""
Generate presentation slides using an LLM agent.

This script uses Claude to read the full topic analysis document and generate
a coherent slide deck with proper structure, removing redundancy and creating
a narrative flow.

Usage:
    python scripts/presentations/generate_slides_agent.py --topic 01
    python scripts/presentations/generate_slides_agent.py --all
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.provider_receipts import (
    gemini_generate_content_stream,
    infer_conference_workspace,
)
from conference_pipeline.model_governance import default_model, validate_selection
from conference_pipeline.presentation.blueprint import (
    PresentationBlueprintValidationError,
    require_approved_blueprint,
    validate_blueprint_file,
)

# Project paths
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
ANALYSIS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "analysis"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
BLUEPRINT_APPROVAL_REQUIRED = "BLUEPRINT_APPROVAL_REQUIRED"
BLUEPRINT_APPROVAL_REQUIRED_MESSAGE = (
    "production draft_generation requires --from-blueprint and approved blueprint_review.json"
)
DRAFT_PROVIDER_REQUIREMENTS = ["env:GEMINI_API_KEY|GOOGLE_API_KEY"]
MODEL_TASK_ID = "presentation.deck_generation"
DEFAULT_MODEL = default_model(MODEL_TASK_ID)

# Topic names mapping
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

# Analysis file names (different from presentation dir names)
ANALYSIS_FILES = {
    "01": "01_tools_and_open_source.md",
    "02": "02_genai_architectures.md",
    "03": "03_data_preparation.md",
    "04": "04_data_attribution.md",
    "05": "05_data_markets.md",
    "06": "06_visual_models.md",
    "07": "07_tabular_models.md",
    "08": "08_language_models.md",
    "09": "09_genai_in_science.md",
    "10": "10_serendipitous_findings.md",
    "11": "11_ethical_problems.md",
    "12": "12_commercial_applications.md",
    "13": "13_industry_insights.md",
    "14": "14_synthetic_data.md",
    "15": "15_whats_happening_at_neurips.md",
    "16": "16_neurips_awards.md",
    "17": "17_llm_scaling_laws_efficiency.md",
    "18": "18_agentic_ai_autonomous_systems.md",
    "19": "19_notable_quotes.md",
    "20": "20_small_language_models.md",
}


SLIDE_GENERATION_PROMPT = '''You are creating a professional presentation from a technical analysis document.

TASK: Read the analysis document and create a slide deck with exactly {target_slides} content slides.

DOCUMENT TO ANALYZE:
{document_content}

PAPERS AVAILABLE FOR REFERENCE:
{papers_summary}

REQUIREMENTS:

1. STRUCTURE (exactly {target_slides} slides):
   - Slide 1: Executive Summary (key takeaways from the document)
   - Slides 2-4: Major Theme 1 (usually 2-4 slides per major theme)
   - Continue with remaining major themes
   - Final 2 slides: Key Takeaways and Future Directions

2. CONTENT RULES:
   - Each slide: 3-5 bullet points maximum
   - Each bullet: 10-20 words, start with action verb or key concept
   - Remove ALL redundancy - if a concept appears multiple times, consolidate
   - Include specific numbers and metrics when available
   - NO generic filler content

3. FOR EACH SLIDE, OUTPUT:
   - slide_type: "content" | "section" | "quote"
   - title: Concise title (5-8 words)
   - bullets: List of 3-5 bullet points
   - source_refs: Which papers/presentations this slide draws from
   - figure_need: What kind of figure would enhance this slide (or "none")
   - speaker_notes: 2-3 sentences of context for the presenter

4. DEDUPLICATION:
   - If the document repeats similar content in different sections, MERGE them
   - Each key concept should appear on exactly ONE slide
   - Prioritize unique insights over repeated information

OUTPUT FORMAT: Return a JSON array of slide objects:
```json
[
  {{
    "slide_num": 1,
    "slide_type": "content",
    "title": "Executive Summary: Data Quality Revolution",
    "bullets": [
      "Dataset distillation achieves 80% data reduction while maintaining performance",
      "New benchmarks (CleanPatrick, PrefCleanBench) expose unsolved cleaning challenges",
      "Shift from static data to optimizable, compressible assets"
    ],
    "source_refs": ["HDD paper", "CleanPatrick benchmark"],
    "figure_need": "overview diagram of data preparation landscape",
    "speaker_notes": "This slide sets up the key themes. Emphasize the paradigm shift."
  }},
  ...
]
```

Generate exactly {target_slides} slides. Focus on unique, high-value insights.
'''


def load_analysis_document(topic_num: str) -> str:
    """Load the analysis document for a topic."""
    filename = ANALYSIS_FILES.get(topic_num)
    if not filename:
        raise ValueError(f"Unknown topic: {topic_num}")

    filepath = ANALYSIS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Analysis file not found: {filepath}")

    return filepath.read_text()


def _format_papers_summary(data: dict) -> str:
    papers = data.get('papers', [])[:10]
    lines = []
    for p in papers:
        has_pdf = "✓ PDF" if p.get('pdf_exists') else "✗ No PDF"
        lines.append(f"- Rank {p['rank']}: {p['title']} ({has_pdf})")

    return "\n".join(lines) if lines else "No papers available."


def load_papers_summary(topic_num: str) -> str:
    """Load a summary of available papers for the topic."""
    topic_name = TOPIC_NAMES.get(topic_num)
    papers_path = PRESENTATIONS_DIR / topic_name / "ranked_papers.json"

    if not papers_path.exists():
        return "No ranked papers available."

    with open(papers_path) as f:
        data = json.load(f)

    return _format_papers_summary(data)


def load_papers_summary_from_dir(output_dir: Path) -> str:
    """Load ranked-paper context from an explicit presentation output directory."""
    papers_path = output_dir / "ranked_papers.json"
    if not papers_path.exists():
        return "No ranked papers available."
    with open(papers_path) as f:
        data = json.load(f)
    return _format_papers_summary(data)


def generate_slides_with_claude(
    topic_num: str,
    target_slides: int = 60,
    verbose: bool = False,
    *,
    document: str | None = None,
    papers: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """Use the registry-approved Gemini deck model to generate slides."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        try:
            import google.generativeai as genai
            from google.generativeai import types
        except ImportError:
            print("Error: google-genai package not installed. Run: pip install google-genai")
            sys.exit(1)

    import os
    import time

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    # Load content (no truncation — send the full document)
    document = document if document is not None else load_analysis_document(topic_num)
    papers = papers if papers is not None else load_papers_summary(topic_num)

    if verbose:
        print(f"  Document length: {len(document)} characters")
        print(f"  Target slides: {target_slides}")

    # Create prompt
    prompt = SLIDE_GENERATION_PROMPT.format(
        target_slides=target_slides,
        document_content=document,
        papers_summary=papers
    )

    selected_model = model or DEFAULT_MODEL
    validate_selection(MODEL_TASK_ID, provider="gemini", model=selected_model)

    # Call Gemini with streaming to avoid timeouts on large responses
    client = genai.Client(api_key=api_key)

    if verbose:
        print(f"  Calling Gemini {selected_model} (streaming)...")

    max_retries = 5
    response_text = ""
    for attempt in range(max_retries):
        try:
            response_text = ""
            stream = gemini_generate_content_stream(client, caller=__file__,
                model=selected_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                )
            )
            char_count = 0
            for chunk in stream:
                if chunk.text:
                    response_text += chunk.text
                    char_count += len(chunk.text)
                    if verbose and char_count % 2000 < len(chunk.text):
                        print(f"    ...received {char_count} chars", flush=True)
            if verbose:
                print(f"    Total response: {len(response_text)} chars")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 10  # 10, 20, 40, 80 seconds
                if verbose:
                    print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(wait_time)
            else:
                raise

    # Extract JSON from response
    import re
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if not json_match:
        raise ValueError("Could not find JSON array in response")

    slides = json.loads(json_match.group())

    if verbose:
        print(f"  Generated {len(slides)} slides")

    return slides


def slides_to_markdown(slides: list[dict], topic_name: str) -> str:
    """Convert slide objects to Markdown format."""
    lines = []

    # Header
    display_name = topic_name.replace('_', ' ').title()
    lines.append(f"# {display_name} at NeurIPS 2025")
    lines.append("")
    lines.append("<!--")
    lines.append(f"GENERATED: {datetime.now().isoformat()}")
    lines.append(f"TOPIC: {topic_name}")
    lines.append("GENERATOR: generate_slides_agent.py (Claude)")
    lines.append("-->")
    lines.append("")
    lines.append("---")
    lines.append("")

    for slide in slides:
        slide_num = slide.get('slide_num', 0)
        slide_type = slide.get('slide_type', 'content')
        title = slide.get('title', 'Untitled')
        bullets = slide.get('bullets', [])
        source_refs = slide.get('source_refs', [])
        figure_need = slide.get('figure_need', 'none')
        speaker_notes = slide.get('speaker_notes', '')

        # Slide header
        lines.append(f"## Slide {slide_num}: {title}")
        lines.append("")

        # Content based on type
        if slide_type == 'quote' and 'quote' in slide:
            lines.append(f"> \"{slide['quote']}\"")
            if 'quote_source' in slide:
                lines.append(f"> — {slide['quote_source']}")
        else:
            for bullet in bullets:
                lines.append(f"- {bullet}")

        lines.append("")

        # Metadata comment
        lines.append("<!--")
        lines.append(f"SLIDE_TYPE: {slide_type}")
        if source_refs:
            lines.append(f"SOURCE_REFS: {', '.join(source_refs)}")
        if figure_need and figure_need != 'none':
            lines.append(f"FIGURE_NEED: {figure_need}")
        if speaker_notes:
            lines.append(f"SPEAKER_NOTES: {speaker_notes}")
        lines.append("-->")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def workspace_slides_to_markdown(
    slides: list[dict],
    *,
    display_title: str,
    presentation_id: str,
    presentation_type: str,
    topic: str | None,
) -> str:
    """Convert workspace-generated slide objects to conference-neutral Markdown."""
    lines = []

    lines.append(f"# {display_title}")
    lines.append("")
    lines.append("<!--")
    lines.append(f"GENERATED: {datetime.now().isoformat()}")
    lines.append(f"PRESENTATION_ID: {presentation_id}")
    lines.append(f"PRESENTATION_TYPE: {presentation_type}")
    if topic is not None:
        lines.append(f"TOPIC: {topic}")
    lines.append("GENERATOR: generate_slides_agent.py (Claude)")
    lines.append("-->")
    lines.append("")
    lines.append("---")
    lines.append("")

    for slide in slides:
        slide_num = slide.get('slide_num', 0)
        slide_type = slide.get('slide_type', 'content')
        title = slide.get('title', 'Untitled')
        bullets = slide.get('bullets', [])
        source_refs = slide.get('source_refs', [])
        figure_need = slide.get('figure_need', 'none')
        speaker_notes = slide.get('speaker_notes', '')
        blueprint_id = slide.get("blueprint_id")
        blueprint_slide_id = slide.get("blueprint_slide_id")
        blueprint_cluster_id = slide.get("blueprint_cluster_id")
        approved_source_labels = slide.get("approved_source_labels", [])
        visual_role = slide.get("visual_role")
        narration_beat = slide.get("narration_beat")

        lines.append(f"## Slide {slide_num}: {title}")
        lines.append("")

        if slide_type == 'quote' and 'quote' in slide:
            lines.append(f"> \"{slide['quote']}\"")
            if 'quote_source' in slide:
                lines.append(f"> — {slide['quote_source']}")
        else:
            for bullet in bullets:
                lines.append(f"- {bullet}")

        lines.append("")
        lines.append("<!--")
        lines.append(f"SLIDE_TYPE: {slide_type}")
        if blueprint_id:
            lines.append(f"BLUEPRINT_ID: {blueprint_id}")
        if blueprint_slide_id:
            lines.append(f"BLUEPRINT_SLIDE_ID: {blueprint_slide_id}")
        if blueprint_cluster_id:
            lines.append(f"BLUEPRINT_CLUSTER_ID: {blueprint_cluster_id}")
        if source_refs:
            lines.append(f"SOURCE_REFS: {', '.join(source_refs)}")
        if approved_source_labels:
            lines.append(f"APPROVED_SOURCE_LABELS: {', '.join(approved_source_labels)}")
        if visual_role:
            lines.append(f"VISUAL_ROLE: {visual_role}")
        if narration_beat:
            lines.append(f"NARRATION_BEAT: {narration_beat}")
        if figure_need and figure_need != 'none':
            lines.append(f"FIGURE_NEED: {figure_need}")
        if speaker_notes:
            lines.append(f"SPEAKER_NOTES: {speaker_notes}")
        lines.append("-->")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def process_topic(topic_num: str, target_slides: int = 60, verbose: bool = False) -> Path:
    """Process a single topic."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name
    topic_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nProcessing: {topic_name}")

    # Generate slides using Claude
    slides = generate_slides_with_claude(topic_num, target_slides, verbose)

    # Convert to Markdown
    markdown = slides_to_markdown(slides, topic_name)

    # Save Markdown
    output_path = topic_dir / "1_draft_slides.md"
    output_path.write_text(markdown)

    # Also save JSON for debugging/processing
    json_path = topic_dir / "slides_structure.json"
    with open(json_path, 'w') as f:
        json.dump(slides, f, indent=2)

    if verbose:
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")
        print(f"  Saved: {json_path.relative_to(PROJECT_ROOT)}")

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
        description="Generate presentation slides using Claude agent"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--display-title", help="Manifest-sourced presentation display title.")
    parser.add_argument("--analysis-md", type=Path, help="Workspace-relative or absolute analysis markdown input.")
    parser.add_argument("--from-blueprint", action="store_true", help="Generate draft slides from an approved presentation blueprint.")
    parser.add_argument("--presentation-blueprint", type=Path, help="Workspace-relative or absolute presentation_blueprint.json input.")
    parser.add_argument("--blueprint-receipt", type=Path, help="Workspace-relative or absolute presentation_blueprint_receipt.json input.")
    parser.add_argument("--blueprint-review", type=Path, help="Workspace-relative or absolute approved blueprint_review.json input.")
    parser.add_argument("--output-dir", type=Path, help="Workspace-relative or absolute deck output directory.")
    parser.add_argument("--output-slides-md", type=Path, help="Workspace-relative or absolute draft slides output path.")
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
                ("--display-title", args.display_title),
                ("--output-dir", args.output_dir),
                ("--output-slides-md", args.output_slides_md),
            )
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if args.from_blueprint:
            missing.extend(
                flag
                for flag, value in (
                    ("--presentation-blueprint", args.presentation_blueprint),
                    ("--blueprint-receipt", args.blueprint_receipt),
                    ("--blueprint-review", args.blueprint_review),
                )
                if value is None
            )
        elif args.analysis_md is None:
            missing.append("--analysis-md")
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")
        if not args.from_blueprint and not args.dry_run:
            parser.error(f"{BLUEPRINT_APPROVAL_REQUIRED}: {BLUEPRINT_APPROVAL_REQUIRED_MESSAGE}")
        if args.presentation_type == "topic-deep-dive" and not args.topic:
            parser.error("--workspace with --presentation-type topic-deep-dive requires --topic")

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        args.display_title = args.display_title.strip()
        path_args = [
            ("output_dir", "--output-dir"),
            ("output_slides_md", "--output-slides-md"),
        ]
        if args.analysis_md is not None:
            path_args.append(("analysis_md", "--analysis-md"))
        if args.from_blueprint:
            path_args.extend(
                [
                    ("presentation_blueprint", "--presentation-blueprint"),
                    ("blueprint_receipt", "--blueprint-receipt"),
                    ("blueprint_review", "--blueprint-review"),
                ]
            )
        for attr, label in path_args:
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr in {"output_dir", "output_slides_md"}:
                _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _workspace_deck_label(args: argparse.Namespace) -> str:
    if args.topic:
        return str(args.topic)
    if args.presentation_id:
        return str(args.presentation_id)
    return str(args.presentation_type)


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "analysis_md": str(args.analysis_md) if args.analysis_md is not None else None,
        "draft_input_mode": "blueprint" if args.from_blueprint else "legacy/mock-only",
        "dry_run": True,
        "output_dir": str(args.output_dir),
        "output_slides_md": str(args.output_slides_md),
        "presentation_blueprint": str(args.presentation_blueprint) if args.presentation_blueprint else None,
        "blueprint_receipt": str(args.blueprint_receipt) if args.blueprint_receipt else None,
        "blueprint_review": str(args.blueprint_review) if args.blueprint_review else None,
        "source_plan_repair_of": _blueprint_source_plan_repair_of(args.presentation_blueprint)
        if args.presentation_blueprint and args.presentation_blueprint.is_file()
        else None,
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "display_title": args.display_title,
        "provider_requirements": DRAFT_PROVIDER_REQUIREMENTS,
        "topic": args.topic,
    }


def _blueprint_source_plan_repair_of(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    value = payload.get("source_plan_repair_of")
    return value if isinstance(value, str) and value.strip() else None


def _load_approved_blueprint(path: Path, review_path: Path, receipt_path: Path | None) -> dict[str, object]:
    try:
        require_approved_blueprint(
            path,
            review_path,
            receipt_path=receipt_path,
            require_live_receipt=receipt_path is not None,
        )
    except PresentationBlueprintValidationError as exc:
        for error in exc.errors:
            print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(2) from exc
    result = validate_blueprint_file(path, fail_on_error=True)
    assert result.blueprint is not None
    return result.blueprint.to_dict()


def _cluster_ids_by_slide(blueprint: dict[str, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cluster in blueprint.get("subtopic_clusters", []):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "")
        for slide_id in cluster.get("slide_bindings", []):
            if cluster_id and isinstance(slide_id, str):
                mapping.setdefault(slide_id, cluster_id)
    return mapping


def _source_labels(source_plan: object) -> list[str]:
    labels: list[str] = []
    if not isinstance(source_plan, list):
        return labels
    for source in source_plan:
        label: object
        if isinstance(source, str):
            label = source
        elif isinstance(source, dict):
            label = source.get("label")
        else:
            label = None
        if isinstance(label, str) and label.strip() and label not in labels:
            labels.append(label.strip())
    return labels


def _title_from_intent(intent: str, fallback: str) -> str:
    cleaned = " ".join(intent.split())
    if not cleaned:
        return fallback
    words = cleaned.split()
    title = " ".join(words[:8]).strip(" .,:;")
    return title or fallback


def _blueprint_to_slide_objects(blueprint: dict[str, object]) -> list[dict[str, object]]:
    blueprint_id = str(blueprint["blueprint_id"])
    cluster_by_slide = _cluster_ids_by_slide(blueprint)
    slides: list[dict[str, object]] = []
    for index, source_slide in enumerate(blueprint.get("slides", []), start=1):
        if not isinstance(source_slide, dict):
            continue
        slide_id = str(source_slide.get("slide_id") or f"slide-{index:03d}")
        intent = str(source_slide.get("intent") or source_slide.get("slide_intent") or "")
        visual_role = str(source_slide.get("visual_role") or "none")
        narration_beat = str(source_slide.get("narration_beat") or "")
        source_refs = _source_labels(source_slide.get("source_plan"))
        bullets = [
            intent,
            f"Evidence: {', '.join(source_refs)}" if source_refs else "",
            f"Visual: {visual_role}" if visual_role and visual_role != "none" else "",
        ]
        slides.append(
            {
                "slide_num": index,
                "slide_type": source_slide.get("slide_type", "content"),
                "title": source_slide.get("title") or _title_from_intent(intent, f"Blueprint Slide {index}"),
                "bullets": [bullet for bullet in bullets if bullet],
                "source_refs": source_refs,
                "figure_need": visual_role,
                "speaker_notes": narration_beat,
                "blueprint_id": blueprint_id,
                "blueprint_slide_id": slide_id,
                "blueprint_cluster_id": cluster_by_slide.get(slide_id, ""),
                "approved_source_labels": source_refs,
                "visual_role": visual_role,
                "narration_beat": narration_beat,
                "slide_intent": intent,
                "estimated_minutes": source_slide.get("estimated_minutes")
                or source_slide.get("timing_estimate_minutes"),
            }
        )
    return slides


def process_workspace_draft(args: argparse.Namespace) -> Path:
    """Process one manifest-backed draft-generation request."""
    if args.from_blueprint:
        blueprint = _load_approved_blueprint(
            args.presentation_blueprint,
            args.blueprint_review,
            args.blueprint_receipt,
        )
        slides = _blueprint_to_slide_objects(blueprint)
    else:
        document = args.analysis_md.read_text(encoding="utf-8")
        papers = load_papers_summary_from_dir(args.output_dir)
        deck_label = _workspace_deck_label(args)
        slides = generate_slides_with_claude(
            deck_label,
            args.slides,
            args.verbose,
            document=document,
            papers=papers,
        )
    markdown = workspace_slides_to_markdown(
        slides,
        display_title=args.display_title,
        presentation_id=args.presentation_id,
        presentation_type=args.presentation_type,
        topic=args.topic,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_slides_md.write_text(markdown, encoding="utf-8")

    json_path = args.output_dir / "slides_structure.json"
    with open(json_path, 'w') as f:
        json.dump(slides, f, indent=2)

    if args.verbose:
        print(f"  Saved: {args.output_slides_md}")
        print(f"  Saved: {json_path}")

    return args.output_slides_md


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # D-QUICK-260715-rz1: establish a workspace-rooted provider_receipt_context
    # so standalone invocations don't fall back to Path.cwd() (repo root);
    # --workspace is optional here for the legacy topic-name code path.
    _receipt_workspace = args.workspace or infer_conference_workspace(".")
    if _receipt_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_receipt_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_slides_agent")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        output_path = process_workspace_draft(args)
        print(f"Generated: {output_path}")
        return 0

    if args.all:
        print(f"Generating slides for all topics (target: {args.slides} slides each)...")
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
