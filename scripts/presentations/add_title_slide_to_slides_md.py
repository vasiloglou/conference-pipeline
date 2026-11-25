#!/usr/bin/env python3
"""Ensure a manifest-backed slide deck starts with a Slide 0 title slide."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - project runtime includes PyYAML
    yaml = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATION_NOTE = (
    "Use decks/<presentation>/... for reusable presentation outputs; "
    "presentations/ is a legacy NeurIPS lane."
)
DEFAULT_SLIDES_FILENAME = "5_draft_slides_with_images.md"
DECK_TITLE_SUFFIX_RE = re.compile(
    r"\s+(?:slide\s+brief|presentation\s+brief|deck\s+brief)\s*$",
    flags=re.IGNORECASE,
)

LEGACY_NEURIPS_TOPICS = [
    ("01_tools_and_open_source", "Tools and Open Source"),
    ("02_genai_architectures", "GenAI Architectures"),
    ("03_data_preparation", "Data Preparation"),
    ("04_data_attribution", "Data Attribution"),
    ("05_data_markets", "Data Markets"),
    ("06_visual_models", "Visual Models"),
    ("07_tabular_models", "Tabular Models"),
    ("08_language_models", "Language Models"),
    ("09_genai_in_science", "GenAI in Science"),
    ("10_serendipitous_findings", "Serendipitous Findings"),
    ("11_ethical_problems", "Ethical Problems"),
    ("12_commercial_applications", "Commercial Applications"),
    ("13_industry_insights", "Industry Insights"),
    ("14_synthetic_data", "Synthetic Data"),
    ("15_whats_happening_at_neurips", "What's Happening at NeurIPS"),
    ("16_neurips_awards", "NeurIPS Awards"),
    ("17_llm_scaling_laws_efficiency", "LLM Scaling Laws & Efficiency"),
    ("18_agentic_ai_autonomous_systems", "Agentic AI & Autonomous Systems"),
    ("19_notable_quotes", "Notable Quotes"),
    ("20_small_language_models", "Small Language Models"),
]


def has_slide_0_title(content: str) -> bool:
    """Return True when the first explicit slide is a title Slide 0."""
    match = re.search(
        r"^## Slide 0:.*?(?=^---\s*$|^## Slide [1-9]\d*:)",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )
    return bool(match and re.search(r"^SLIDE_TYPE:\s*title\s*$", match.group(0), re.MULTILINE))


def remove_slide_0(content: str) -> str:
    """Remove an existing Slide 0 block from the markdown."""
    pattern = r"^## Slide 0:.*?(?=^## Slide [1-9]\d*:)"
    return re.sub(pattern, "", content, count=1, flags=re.MULTILINE | re.DOTALL)


def clean_display_title(display_title: str) -> str:
    """Return a deck-facing title suitable for a large cover slide."""
    title = re.sub(r"\s+", " ", display_title).strip()
    title = DECK_TITLE_SUFFIX_RE.sub("", title).strip()
    return title or display_title.strip()


def _speaker_title(display_title: str, conference_label: str) -> str:
    title = clean_display_title(display_title)
    if conference_label and conference_label.lower() in title.lower():
        return title
    return f"{title} at {conference_label}".strip()


def create_slide_0_block(display_title: str, conference_label: str) -> str:
    """Create the canonical title slide block used before Reveal.js render."""
    clean_title = clean_display_title(display_title)
    full_title = _speaker_title(clean_title, conference_label)
    speaker_title = clean_title.lower()
    return f"""## Slide 0: {full_title}

# {clean_title}

*{conference_label}*

<!--
SECTION: Opening
SLIDE_TYPE: title
SPEAKER_NOTES: Welcome. This presentation covers {speaker_title} at {conference_label}.
-->

---

"""


def fix_h1_title(content: str, display_title: str, conference_label: str) -> str:
    """Normalize the document H1 so rendered metadata matches Slide 0."""
    proper_title = _speaker_title(display_title, conference_label)
    if re.search(r"^#\s+.+$", content, flags=re.MULTILINE):
        return re.sub(r"^#\s+.+$", f"# {proper_title}", content, count=1, flags=re.MULTILINE)
    return f"# {proper_title}\n\n{content.lstrip()}"


def process_slides(
    slides_md: Path,
    *,
    display_title: str,
    conference_label: str,
    output_slides_md: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Add or normalize Slide 0. Returns a receipt-friendly result dict."""
    output_path = output_slides_md or slides_md
    original = slides_md.read_text(encoding="utf-8")
    content = remove_slide_0(original)
    content = fix_h1_title(content, display_title, conference_label)

    match = re.search(r"^## Slide [1-9]\d*:", content, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"No numbered slides found in {slides_md}")

    updated = content[: match.start()] + create_slide_0_block(display_title, conference_label) + content[match.start() :]
    changed = updated != original or output_path.resolve() != slides_md.resolve()
    if changed and not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(updated, encoding="utf-8")
    return {
        "slides_md": str(slides_md),
        "output_slides_md": str(output_path),
        "changed": changed,
        "dry_run": dry_run,
        "has_slide_0_title": has_slide_0_title(updated),
        "display_title": display_title,
        "conference_label": conference_label,
    }


def infer_display_title(slides_md: Path, presentation_id: str | None) -> str:
    try:
        content = slides_md.read_text(encoding="utf-8")
    except OSError:
        content = ""
    title_match = re.search(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    if title_match:
        title = re.sub(r"\s+at\s+.+$", "", title_match.group(1).strip())
        if title:
            return clean_display_title(title)
    if presentation_id:
        return re.sub(r"^\d{2}_", "", presentation_id).replace("_", " ").replace("-", " ").title()
    return slides_md.parent.name.replace("_", " ").replace("-", " ").title()


def infer_conference_label(workspace: Path | None) -> str:
    if workspace is None:
        return "NeurIPS 2025"
    config_path = workspace / "config.yaml"
    if yaml is not None and config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            payload = {}
        conference = payload.get("conference") if isinstance(payload, dict) else None
        name = conference.get("name") if isinstance(conference, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    return workspace.name.replace("_", " ").replace("-", " ").title()


def _workspace_relative_path(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must stay inside the workspace: {resolved}") from exc
    if "presentations" in relative.parts:
        raise argparse.ArgumentTypeError(MIGRATION_NOTE)
    return resolved


def _legacy_neurips_run(args: argparse.Namespace) -> int:
    """Preserve the historical no-arg NeurIPS maintenance mode."""
    decks_dir = PROJECT_ROOT / "conferences" / "neurips2025" / "decks"
    results: list[dict[str, object]] = []
    for topic_dir_name, display_title in LEGACY_NEURIPS_TOPICS:
        slides_md = decks_dir / topic_dir_name / DEFAULT_SLIDES_FILENAME
        if not slides_md.exists():
            results.append({"slides_md": str(slides_md), "status": "missing"})
            continue
        results.append(
            process_slides(
                slides_md,
                display_title=display_title,
                conference_label=args.conference_label or "NeurIPS 2025",
                dry_run=args.dry_run,
            )
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure a deck has canonical Slide 0 title metadata")
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--topic", help="Selected topic metadata for manifest-backed runs.")
    parser.add_argument("--slides-md", type=Path, help="Input markdown slides file.")
    parser.add_argument("--output-slides-md", type=Path, help="Output markdown slides file. Defaults to in-place.")
    parser.add_argument("--display-title", help="Human-facing presentation title.")
    parser.add_argument("--conference-label", help="Conference label for the subtitle, e.g. ICLR 2025.")
    parser.add_argument("--dry-run", action="store_true", help="Report the title-slide update without writing.")
    parser.add_argument(
        "--legacy-neurips-all",
        action="store_true",
        help="Maintain historical NeurIPS topic decks when no manifest scope is provided.",
    )
    args = parser.parse_args(argv)

    if args.slides_md is None:
        if args.legacy_neurips_all:
            return args
        parser.error("--slides-md is required")

    if args.workspace is not None:
        missing = [
            flag
            for flag, value in (
                ("--presentation-id", args.presentation_id),
                ("--presentation-type", args.presentation_type),
                ("--slides-md", args.slides_md),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")
        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        try:
            args.slides_md = _workspace_relative_path(args.slides_md, workspace, label="--slides-md")
            if args.output_slides_md is not None:
                args.output_slides_md = _workspace_relative_path(
                    args.output_slides_md,
                    workspace,
                    label="--output-slides-md",
                )
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.slides_md is None:
        return _legacy_neurips_run(args)

    slides_md = args.slides_md.expanduser().resolve()
    if not slides_md.exists():
        print(f"Error: {slides_md} not found", file=sys.stderr)
        return 1
    display_title = args.display_title or infer_display_title(slides_md, args.presentation_id)
    conference_label = args.conference_label or infer_conference_label(args.workspace)
    result = process_slides(
        slides_md,
        display_title=display_title,
        conference_label=conference_label,
        output_slides_md=args.output_slides_md,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
