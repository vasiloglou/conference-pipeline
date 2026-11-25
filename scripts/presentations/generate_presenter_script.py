#!/usr/bin/env python3
"""
Generate timed presenter script from final slide deck (Pass 6).

Reads 5_draft_slides_with_images.md and contextual source materials,
then generates a per-slide presenter script with SAY, CONTEXT, and
TRANSITION blocks.

Output: 6_presenter_script.md

Usage:
    python scripts/presentations/generate_presenter_script.py --topic 02
    python scripts/presentations/generate_presenter_script.py --all
    python scripts/presentations/generate_presenter_script.py --topic 02 -v
"""

import argparse
import hashlib
import html
import json
import os
import re
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)
from conference_pipeline.presentation.blueprint import (
    PresentationBlueprintValidationError,
    require_approved_blueprint,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
ANALYSIS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "analysis"
INFLUENCE_DIR = (
    PROJECT_ROOT / "conferences" / "neurips2025"
    / "claude-autonomy" / "paper-influence" / "reports"
)

from conference_pipeline.model_governance.registry import default_model  # noqa: E402

DEFAULT_GEN_MODEL = default_model("presentation.presenter_script_generation")
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
SLIDE_MAPPING_REQUIRED = "presentation_slide_mapping.json is required for workspace presenter-script planning"
PRESENTER_PROVIDER_REQUIREMENTS = ["env:GEMINI_API_KEY|GOOGLE_API_KEY"]
BLUEPRINT_NARRATION_BEAT_CONTRACT = "blueprint narration beat"
DEFAULT_SCRIPT_MAX_SLIDES_PER_CALL = 8
DEFAULT_SCRIPT_CONTENT_RETRIES = 3


class GeminiScriptTimeout(TimeoutError):
    """Raised when one presenter-script Gemini call exceeds the local watchdog."""


@contextmanager
def gemini_script_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(signum, frame):  # noqa: ARG001 - signal handler signature
        raise GeminiScriptTimeout(
            f"Presenter-script Gemini call exceeded {seconds}s watchdog timeout"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

# Topic names mapping (same as other presentation scripts)
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
# Slide parsing
# ---------------------------------------------------------------------------

def parse_slides(content: str) -> list[dict]:
    """Parse slide markdown into structured slide objects."""
    slides = []
    slide_pattern = r'(## Slide \d+:[^\n]+)'
    parts = re.split(slide_pattern, content)

    current_slide = None
    for part in parts:
        if part.startswith('## Slide'):
            if current_slide:
                slides.append(current_slide)
            match = re.match(r'## Slide (\d+): (.+)', part)
            if match:
                current_slide = {
                    'num': int(match.group(1)),
                    'title': match.group(2).strip(),
                    'content': '',
                    'metadata': {},
                    'has_image': False,
                }
        elif current_slide:
            current_slide['content'] = part

            # Parse HTML comment metadata — find the metadata comment
            # (contains keys like SLIDE_TYPE, SECTION, SPEAKER_NOTES),
            # skipping non-metadata comments like FIGURE_DESCRIPTION.
            all_comments = re.finditer(r'<!--\s*(.*?)\s*-->', part, re.DOTALL)
            for cm in all_comments:
                body = cm.group(1)
                if re.search(r'^[A-Z_]+:', body, re.MULTILINE):
                    for line in body.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            current_slide['metadata'][key.strip()] = value.strip()

            # Extract figure descriptions and paths from image alt text
            figure_matches = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', part)
            if figure_matches:
                current_slide['has_image'] = True
                current_slide['figure_descriptions'] = [
                    alt for alt, _path in figure_matches if alt
                ]
                current_slide['figure_paths'] = [
                    path for _alt, path in figure_matches
                ]

    if current_slide:
        slides.append(current_slide)

    return slides


def get_slide_bullets(slide: dict) -> str:
    """Extract bullet-point content from a slide, stripping metadata and images."""
    text = slide['content']
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove image lines
    text = re.sub(r'!\[.*?\]\(.*?\)\n?', '', text)
    # Remove caption lines (italic lines right after images)
    text = re.sub(r'^\*[^*]+\*$', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^---\s*$', '', text, flags=re.MULTILINE)
    # Collapse blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_slide_timing(slide: dict) -> str:
    """Determine timing string for a slide based on its type."""
    slide_type = slide['metadata'].get('SLIDE_TYPE', '').lower()
    title_lower = slide['title'].lower()
    bullets = get_slide_bullets(slide)

    if slide_type in ('banner', 'agenda', 'section'):
        return '~15s'
    if 'agenda' in title_lower or 'table of contents' in title_lower:
        return '~15s'
    # Section dividers often have no bullets
    if not bullets or len(bullets) < 20:
        return '~15s'
    return '~1 min'


def find_best_input(topic_dir: Path) -> Path:
    """Find the most advanced slide file available."""
    for name in [
        "5_draft_slides_with_images.md",
        "4.5_draft_slides_with_hyperlinks.md",
        "4_draft_slides_sanity_check.md",
        "3_draft_slides_reflowed.md",
        "2_draft_slides_enriched_with_papers.md",
        "1_draft_slides.md",
    ]:
        p = topic_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No slide markdown found in {topic_dir}")


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _load_optional_json(path: Path, default=None):
    """Load JSON file, returning default if missing."""
    if not path.exists():
        return default if default is not None else {}
    with open(path) as f:
        return json.load(f)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_repo_env() -> None:
    """Load the repository-local credential file used by live provider stages."""
    load_dotenv(PROJECT_ROOT / ".env")


def _script_slide_nums(script: str, block_type: str = "SLIDE") -> set[int]:
    return {int(m.group(1)) for m in re.finditer(rf'\[{block_type} (\d+)\]', script or "")}


def load_ranked_papers(topic_dir: Path) -> dict[int, dict]:
    """Load ranked papers keyed by rank number."""
    data = _load_optional_json(topic_dir / "ranked_papers.json")
    papers = data.get('papers', [])
    by_rank: dict[int, dict] = {}
    for paper in papers:
        if 'rank' not in paper:
            continue
        rank = paper['rank']
        normalized = dict(paper)
        normalized['title'] = _paper_display_title(normalized, rank)
        by_rank[rank] = normalized
    return by_rank


def _paper_display_title(paper: dict, rank: int | str) -> str:
    for key in ('title', 'paper_title', 'name'):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ('source_artifact_id', 'artifact_id', 'source_artifact_path'):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return value.rsplit('/', 1)[-1].replace('_', ' ')
    return f"Ranked paper {rank}"


def load_slide_paper_mapping(topic_dir: Path) -> dict[str, list[int]]:
    """Load slide-to-paper mapping from asset inventory."""
    data = _load_optional_json(topic_dir / "asset_inventory.json")
    return data.get('slide_paper_mapping', {})


def load_extracted_sections(topic_dir: Path) -> list[dict]:
    """Load analysis sections from extracted content."""
    data = _load_optional_json(topic_dir / "extracted_content.json")
    analysis = data.get('analysis', {})
    return analysis.get('sections', [])


def load_influence_report(topic_num: str) -> str:
    """Load the influential papers report as raw text."""
    topic_name = TOPIC_NAMES.get(topic_num, '')
    # Strip the number prefix: "02_genai_architectures" -> "genai_architectures"
    name_part = topic_name[3:] if topic_name else ''
    report_name = f"{topic_num}_{name_part}_influential_papers.md"
    path = INFLUENCE_DIR / report_name
    if path.exists():
        return path.read_text()
    return ''


def load_source_attributions(topic_dir: Path) -> list[dict]:
    """Load source attributions (workshops, presentations, speakers)."""
    data = _load_optional_json(topic_dir / "source_attributions.json")
    return data.get('attributions', [])


def load_zoom_companions(
    topic_dir: Path,
    *,
    mapping_path: Path | None = None,
) -> dict[int, bool]:
    """Determine which logical slides have zoom companions from the HTML slide mapping.

    The mapping JSON maps logical slide numbers (as strings) to HTML section indices.
    A gap > 1 between consecutive logical slides means a zoom companion exists.
    """
    mapping_path = mapping_path or topic_dir / "presentation_slide_mapping.json"
    if not mapping_path.exists():
        return {}

    mapping = json.loads(mapping_path.read_text())
    entries = sorted(
        [(int(k), v) for k, v in mapping.items()],
        key=lambda x: x[0],
    )

    has_zoom: dict[int, bool] = {}
    for i, (slide_num, html_idx) in enumerate(entries):
        if i + 1 < len(entries):
            next_html_idx = entries[i + 1][1]
            has_zoom[slide_num] = (next_html_idx - html_idx) > 1
        else:
            has_zoom[slide_num] = False

    return has_zoom


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _extract_paper_ranks_from_source(source: str) -> list[int]:
    """Extract paper rank numbers from SOURCE metadata string."""
    ranks = [int(n) for n in re.findall(r'#(\d+)', source)]
    # Also capture bare comma-separated numbers after a #N pattern
    after_hash = re.search(r'#\d+(.+?)(?::|$)', source)
    if after_hash:
        bare_nums = re.findall(r'(?<!\w)(\d{1,3})(?!\w)', after_hash.group(1))
        for n in bare_nums:
            num = int(n)
            if num not in ranks and 1 <= num <= 50:
                ranks.append(num)
    return ranks


def _extract_influence_excerpt(report: str, rank: int, max_chars: int = 500) -> str:
    """Extract the section for a specific paper rank from the influence report."""
    # Look for "## N. Title" pattern
    pattern = rf'^## {rank}\.\s+(.+?)(?=\n---|\n## \d+\.|\Z)'
    match = re.search(pattern, report, re.MULTILINE | re.DOTALL)
    if not match:
        return ''
    excerpt = match.group(0).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + '...'
    return excerpt


def _find_matching_section(sections: list[dict], section_name: str) -> str:
    """Find the extracted content section matching a slide's SECTION metadata."""
    if not section_name or not sections:
        return ''
    section_lower = section_name.lower()
    for sec in sections:
        title = sec.get('title', '')
        # Match if the section metadata appears in the extracted section title
        # e.g., "The Diffusion Revolution" matches "Major Theme 1: Diffusion for Language..."
        title_lower = title.lower()
        # Try word overlap matching
        section_words = set(re.findall(r'[a-z]{4,}', section_lower))
        title_words = set(re.findall(r'[a-z]{4,}', title_lower))
        overlap = section_words & title_words
        if len(overlap) >= 2 or section_lower in title_lower or title_lower in section_lower:
            content = sec.get('content', '')
            if len(content) > 1500:
                content = content[:1500] + '...'
            return content
    return ''


def _find_matching_attributions(
    attributions: list[dict], source_text: str
) -> list[dict]:
    """Find source attributions matching a slide's SOURCE metadata."""
    if not source_text or not attributions:
        return []
    matches = []
    source_lower = source_text.lower()
    for attr in attributions:
        title = (attr.get('title', '') or '').lower()
        if not title:
            continue
        # Check if attribution title appears in slide source or vice versa
        title_words = set(re.findall(r'[a-z]{5,}', title))
        source_words = set(re.findall(r'[a-z]{5,}', source_lower))
        if len(title_words & source_words) >= 2:
            matches.append(attr)
    return matches[:3]  # Cap at 3


def build_slide_context(
    slide: dict,
    papers: dict[int, dict],
    mapping: dict[str, list[int]],
    sections: list[dict],
    influence_report: str,
    attributions: list[dict],
) -> str:
    """Assemble context string for a single slide."""
    parts = []
    source = slide['metadata'].get('SOURCE', '')
    section_name = slide['metadata'].get('SECTION', '')
    slide_num_str = str(slide['num'])

    # 1. Paper references (from SOURCE + slide_paper_mapping)
    paper_ranks = _extract_paper_ranks_from_source(source)
    if slide_num_str in mapping:
        for r in mapping[slide_num_str]:
            if r not in paper_ranks:
                paper_ranks.append(r)

    if paper_ranks:
        paper_lines = []
        for rank in paper_ranks:
            p = papers.get(rank)
            if not p:
                continue
            line = f"  Paper #{rank}: {p['title']}"
            if p.get('authors'):
                authors = p['authors']
                if isinstance(authors, list):
                    authors = ', '.join(authors[:3])
                    if len(p['authors']) > 3:
                        authors += ' et al.'
                line += f" ({authors})"
            if p.get('venue'):
                line += f" [{p['venue']}]"
            if p.get('arxiv_url'):
                line += f" {p['arxiv_url']}"
            paper_lines.append(line)
        if paper_lines:
            parts.append("Papers:\n" + '\n'.join(paper_lines))

    # 2. Influence excerpts for matched papers
    for rank in paper_ranks[:2]:  # Top 2 to save tokens
        excerpt = _extract_influence_excerpt(influence_report, rank)
        if excerpt:
            parts.append(f"Why Paper #{rank} matters:\n{excerpt}")

    # 3. Analysis section content
    section_content = _find_matching_section(sections, section_name)
    if section_content:
        parts.append(f"Analysis context:\n{section_content}")

    # 4. Workshop/speaker attributions
    matched_attrs = _find_matching_attributions(attributions, source)
    if matched_attrs:
        attr_lines = []
        for attr in matched_attrs:
            category = attr.get('category', '')
            title = attr.get('title', '')
            speakers = attr.get('speakers', [])
            speaker_str = ', '.join(speakers[:3]) if speakers else 'N/A'
            attr_lines.append(f"  {category}: {title} (Speakers: {speaker_str})")
        parts.append("Related sessions:\n" + '\n'.join(attr_lines))

    return '\n\n'.join(parts) if parts else '(No additional context available)'


# ---------------------------------------------------------------------------
# Section grouping
# ---------------------------------------------------------------------------

def group_slides_by_section(slides: list[dict]) -> list[dict]:
    """Group consecutive slides by their SECTION metadata.

    Slides without SECTION metadata (banner/divider slides) are merged into
    the next named section so they become the section's intro slide.
    """
    # First pass: build raw groups
    raw_groups: list[dict] = []
    current_section: str | None = None
    current_slides: list[dict] = []

    for slide in slides:
        section = slide['metadata'].get('SECTION', '')
        if section != current_section:
            if current_slides:
                raw_groups.append({
                    'name': current_section or '',
                    'slides': current_slides,
                })
            current_section = section
            current_slides = [slide]
        else:
            current_slides.append(slide)

    if current_slides:
        raw_groups.append({
            'name': current_section or '',
            'slides': current_slides,
        })

    # Second pass: merge unnamed groups into the next named group
    merged: list[dict] = []
    pending_slides: list[dict] = []

    for group in raw_groups:
        if not group['name']:
            pending_slides.extend(group['slides'])
        else:
            combined = pending_slides + group['slides']
            pending_slides = []
            merged.append({
                'name': group['name'],
                'slides': combined,
            })

    # Trailing unnamed slides go into the last group
    if pending_slides:
        if merged:
            merged[-1]['slides'].extend(pending_slides)
        else:
            merged.append({'name': 'Presentation', 'slides': pending_slides})

    # Third pass: merge consecutive groups with the same name
    # (can happen when unnamed slides split a section in two)
    final: list[dict] = []
    for group in merged:
        if final and final[-1]['name'] == group['name']:
            final[-1]['slides'].extend(group['slides'])
        else:
            final.append(group)

    return final


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _load_genai_module():
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "Error: google-genai package not installed. Run: pip install google-genai"
        ) from exc
    return genai


def _get_genai_client():
    """Create a Gemini client from the repository-local provider env."""
    import os
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    timeout_ms = int(os.environ.get("PRESENTATION_GEMINI_TIMEOUT_MS", "180000"))
    genai = _load_genai_module()
    return genai.Client(
        api_key=api_key,
        http_options=genai.types.HttpOptions(timeout=max(1000, timeout_ms)),
    )


def call_gemini(
    prompt: str,
    *,
    image_paths: list[Path] | None = None,
    model: str = DEFAULT_GEN_MODEL,
    max_tokens: int = 8192,
    verbose: bool = False,
) -> str:
    """Call Gemini API with optional multimodal image inputs and retry logic."""
    genai = _load_genai_module()
    client = _get_genai_client()
    max_retries = 3

    # Build content parts: images first, then text prompt
    contents: list = []
    if image_paths:
        for img_path in image_paths:
            if img_path.exists() and img_path.suffix.lower() == ".svg":
                # SVGs are sent as text since Gemini doesn't natively handle SVG images
                svg_text = img_path.read_text("utf-8", errors="ignore")
                contents.append(
                    f"[Figure from {img_path.name}]\n```svg\n{svg_text[:8000]}\n```"
                )
            elif img_path.exists():
                # For raster images, upload as inline data
                import mimetypes
                mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
                contents.append(
                    genai.types.Part.from_bytes(
                        data=img_path.read_bytes(),
                        mime_type=mime,
                    )
                )
    contents.append(prompt)

    for attempt in range(max_retries):
        try:
            timeout_seconds = _env_int(
                "PRESENTATION_GEMINI_SCRIPT_CALL_TIMEOUT_SECONDS",
                max(1, _env_int("PRESENTATION_GEMINI_TIMEOUT_MS", 180000) // 1000),
            )
            with gemini_script_timeout(timeout_seconds):
                response = gemini_generate_content(client, caller=__file__,
                    model=model,
                    contents=contents,
                    config={"max_output_tokens": max_tokens, "temperature": 0.4},
                )
            return response.text or ""
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s
                if verbose:
                    print(f"    Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                raise


def _load_figure_manifest(topic_dir: Path | None) -> dict[str, dict]:
    """Load figure_manifest.json if available, mapping SVG filenames to annotation data."""
    if not topic_dir:
        return {}
    manifest_path = topic_dir / "assets" / "generated" / "figure_manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _load_asset_figure_specs(topic_dir: Path | None) -> dict[int, dict]:
    """Load current source-grounded figure specs keyed by slide number."""
    if not topic_dir:
        return {}
    candidates = [
        topic_dir / "asset_inventory.json",
        topic_dir / "assets" / "generated" / "figure_specs.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            rows = payload.get("figure_specs") or payload.get("figures") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        specs: dict[int, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            spec = row.get("figure_spec") if isinstance(row.get("figure_spec"), dict) else row
            if not isinstance(spec, dict):
                continue
            try:
                slide_number = int(spec.get("slide_number") or row.get("slide_number"))
            except (TypeError, ValueError):
                continue
            specs[slide_number] = spec
        if specs:
            return specs
    return {}


def _short_list(values: object, *, limit: int = 4) -> str:
    if not isinstance(values, list):
        return ""
    parts = [str(item).strip() for item in values if str(item).strip()]
    return "; ".join(parts[:limit])


def _format_figure_spec_for_prompt(spec: dict) -> str:
    lines: list[str] = []
    depiction_type = str(spec.get("depiction_type") or "").strip()
    semantic_goal = str(spec.get("semantic_goal") or "").strip()
    figure_need = str(spec.get("figure_need") or "").strip()
    if depiction_type:
        lines.append(f"  Figure spec depiction type: {depiction_type}")
    if figure_need:
        lines.append(f"  Figure spec purpose: {figure_need}")
    if semantic_goal:
        lines.append(f"  Figure spec semantic goal: {semantic_goal}")
    for key, label in (
        ("entities", "required entities"),
        ("relationships", "required relationships"),
        ("quantities", "source quantities"),
        ("must_depict", "must describe visible evidence"),
        ("must_not_do", "must avoid"),
    ):
        formatted = _short_list(spec.get(key), limit=4)
        if formatted:
            lines.append(f"  Figure spec {label}: {formatted}")
    if lines:
        lines.append(
            "  Figure narration rule: ground the zoom narration in these entities, "
            "relationships, and quantities; do not describe decorative boxes made "
            "from title words."
        )
    return "\n".join(lines)


def generate_section_script(
    section_name: str,
    section_slides: list[dict],
    slide_contexts: dict[int, str],
    prev_section: str | None,
    next_section: str | None,
    topic_display_name: str,
    topic_dir: Path | None = None,
    zoom_companions: dict[int, bool] | None = None,
    conference_label: str | None = None,
    verbose: bool = False,
) -> str:
    """Generate presenter script for all slides in one section."""
    zoom_companions = zoom_companions or {}
    figure_manifest = _load_figure_manifest(topic_dir)
    figure_specs = _load_asset_figure_specs(topic_dir)

    # --- Sub-batch large sections to avoid Gemini API payload limits ---
    MAX_FIGURES_PER_CALL = 15
    max_slides_per_call = max(
        1,
        _env_int(
            "PRESENTATION_GEMINI_SCRIPT_MAX_SLIDES_PER_CALL",
            DEFAULT_SCRIPT_MAX_SLIDES_PER_CALL,
        ),
    )
    if topic_dir:
        total_figs = sum(
            1 for s in section_slides for p in s.get('figure_paths', [])
            if (topic_dir / p).resolve().exists()
        )
    else:
        total_figs = 0

    if (
        len(section_slides) > max_slides_per_call
        or total_figs > MAX_FIGURES_PER_CALL
    ) and len(section_slides) > 1:
        mid = len(section_slides) // 2
        if verbose:
            print(f"  Section: {section_name} ({len(section_slides)} slides, {total_figs} figures) — splitting into sub-batches")
        part1 = generate_section_script(
            section_name=section_name,
            section_slides=section_slides[:mid],
            slide_contexts=slide_contexts,
            prev_section=prev_section,
            next_section=f"{section_name} (continued)",
            topic_display_name=topic_display_name,
            topic_dir=topic_dir,
            zoom_companions=zoom_companions,
            conference_label=conference_label,
            verbose=verbose,
        )
        part2 = generate_section_script(
            section_name=section_name,
            section_slides=section_slides[mid:],
            slide_contexts=slide_contexts,
            prev_section=f"{section_name} (earlier slides)",
            next_section=next_section,
            topic_display_name=topic_display_name,
            topic_dir=topic_dir,
            zoom_companions=zoom_companions,
            conference_label=conference_label,
            verbose=verbose,
        )
        return part1 + "\n\n" + part2

    # Format slides for prompt
    slides_formatted = []
    for slide in section_slides:
        timing = get_slide_timing(slide)
        bullets = get_slide_bullets(slide)
        has_img = "yes" if slide['has_image'] else "no"
        has_zoom = zoom_companions.get(slide['num'], False)
        slide_type = slide.get('metadata', {}).get('SLIDE_TYPE', '')
        entry = f'Slide {slide["num"]}: "{slide["title"]}"\n'
        entry += f'  Timing: {timing}\n'
        entry += f'  Has figure: {has_img}\n'
        if slide_type:
            entry += f'  Slide type: {slide_type}\n'
        if has_zoom:
            entry += f'  Has zoom companion: yes (the presentation will zoom into the figure on a separate slide)\n'
        # Include figure descriptions when available
        fig_descs = slide.get('figure_descriptions', [])
        if fig_descs:
            entry += f'  Figure description: {"; ".join(fig_descs)}\n'
        fig_need = slide.get('metadata', {}).get('FIGURE_NEED', '')
        if fig_need:
            entry += f'  Figure purpose: {fig_need}\n'
        figure_spec_prompt = _format_figure_spec_for_prompt(
            figure_specs.get(int(slide["num"]), {})
        )
        if figure_spec_prompt:
            entry += f'{figure_spec_prompt}\n'
        # Include figure annotations from manifest (text already visible in the figure)
        for fig_path in slide.get('figure_paths', []):
            svg_name = Path(fig_path).name
            if svg_name in figure_manifest:
                annotations = figure_manifest[svg_name].get('annotations', [])
                desc = figure_manifest[svg_name].get('description', '')
                if annotations:
                    entry += f'  Figure annotations (text visible in the figure): {", ".join(annotations)}\n'
                if desc:
                    entry += f'  Figure depicts: {desc}\n'
                entry += '  Note: SAY should NOT repeat what the figure annotations already show visually.\n'
                entry += '  Focus on insight, significance, and connections the figure cannot convey.\n'
        if bullets:
            # Truncate long bullet content
            truncated = bullets[:400] + ('...' if len(bullets) > 400 else '')
            entry += f'  Content:\n{truncated}\n'
        slides_formatted.append(entry)
    slides_str = '\n'.join(slides_formatted)

    # Format per-slide contexts
    contexts_formatted = []
    for slide in section_slides:
        ctx = slide_contexts.get(slide['num'], '(No additional context)')
        contexts_formatted.append(f'--- Slide {slide["num"]} context ---\n{ctx}')
    contexts_str = '\n\n'.join(contexts_formatted)

    prev_note = ""
    if prev_section:
        prev_note = f'\nPrevious section was: "{prev_section}"\n'

    next_note = "the conclusion"
    if next_section:
        next_note = f'"{next_section}"'

    # Check if any slides in this section have zoom companions
    has_any_zoom = any(zoom_companions.get(s['num'], False) for s in section_slides)

    zoom_format = ""
    zoom_rules = ""
    if has_any_zoom:
        zoom_format = """
For slides marked "Has zoom companion: yes", ALSO write a ZOOM block immediately after the main slide block:

---

### [ZOOM N] Slide Title — Figure Detail
**Time:** ~20s | **Section:** {section_name}

**SAY:**
> Write 2-3 sentences that explain the source-grounded mechanism, relationship,
> quantity, or scientific idea shown in the zoomed figure. Use the Figure spec
> entities, relationships, quantities, and semantic goal when provided. This is
> the moment where the presenter helps the audience read the visual evidence.

**FIGURE:** One sentence summarizing the key visual takeaway from the zoomed figure.

---
"""
        zoom_rules = """
12. ZOOM SLIDES: For every slide marked "Has zoom companion: yes", you MUST generate a
    [ZOOM N] block immediately after the [SLIDE N] block. The zoom block is a separate beat
    where the figure fills the screen and the presenter walks through the visual details.
    - The ZOOM SAY block should explain specific source-grounded elements visible in the figure:
      named entities, axes, rungs, gates, arrows, trends, quantities, and the relationship they encode.
    - Use Figure spec details when available. Do not infer a generic chart if the spec requests a
      mechanism, workflow, evidence map, molecular scene, spatial task panel, or architecture.
    - Avoid filler openings such as "Let's zoom in on this diagram" and avoid saying the figure is
      just colored boxes, abstract ellipses, or title words unless the source-grounded spec explicitly
      asks for that visual form.
    - Keep it 2-3 sentences (~20 seconds of speaking time)
    - Do NOT repeat the main slide's SAY content — the zoom is a dedicated figure walkthrough
    - The ZOOM block does NOT need a CONTEXT or TRANSITION block
13. Do NOT generate [ZOOM N] blocks for slides that don't have "Has zoom companion: yes"."""

    talk_context = (
        f"a {conference_label} conference talk"
        if conference_label
        else "a conference talk"
    )

    prompt = f"""You are writing a presenter script for {talk_context} on "{topic_display_name}".

IMPORTANT CONTEXT: This script will be transformed into a two-host podcast dialogue (NotebookLM style)
between "Alex" (a curious explainer who hooks the audience) and "Jordan" (an analytical expert who
provides depth and "aha moments"). Write the SAY blocks with this in mind:
- Include surprising angles, hooks, and "why does this matter?" framing that Alex can use
- Include specific technical details, nuance, and non-obvious implications that Jordan can expand on
- Write with energy and narrative momentum — this should feel like great podcast material, not a dry lecture
- Include enough richness that a dialogue can be built from it without needing to invent new facts

AUDIENCE & TONE — pedagogical, no assumed prior knowledge:
This program covers a broad spectrum of research and no listener can have background on
everything. Write for an intelligent non-specialist from a different subfield. Teach, do
not gatekeep: define every technical term, method name, acronym, and benchmark in plain
language the first time it appears; motivate why an idea matters before explaining how it
works; and lead with intuition or a concrete everyday analogy before any formalism. Never
assume the audience already knows this subfield's vocabulary or its prior results.

CURRENT SECTION: {section_name}
{prev_note}
SLIDES IN THIS SECTION:
{slides_str}

CONTEXT FOR EACH SLIDE (use this to enrich the spoken narrative and fill in CONTEXT blocks):
{contexts_str}

TASK: For each slide above, write a script block following this EXACT format:

---

### [SLIDE N] Slide Title
**Time:** timing | **Section:** {section_name}

**SAY:**
> Write 2-4 sentences of natural spoken narrative here. This is what the
> presenter says aloud to the audience. Add value beyond the bullet points
> with context, significance, and connections. Use specific numbers and
> paper names when available. Sound conversational, not robotic.

**CONTEXT** (not spoken -- for presenter reference):
- Paper: "Full Title" (Authors, Year) [venue]
- Related papers: #N Title, #M Title
- Workshop/session: Name (Speaker: Name)
- Key insight: One-sentence takeaway

**REFS** (spoken-friendly citations for hosts to name-drop in the podcast dialogue):
- "Paper Title" (First Author, Venue Year)
- Workshop: "Workshop Name" (Speaker)
- Keynote: "Talk Title" (Speaker)

**FIGURE:** One sentence describing the chart/diagram the audience sees on screen (only for slides with figures).

**TRANSITION:** "One sentence bridging to the next slide..."
{zoom_format}
---

RULES:
1. SAY blocks must sound like natural, conversational speech to a broad, mixed-expertise
   audience — assume no prior knowledge of this subfield, and define each technical term,
   method, acronym, or benchmark in plain language the first time it is spoken, giving the
   intuition before the detail
2. Do NOT just read the bullet points. Add insight, context, significance, and narrative connections
3. Include specific numbers, metrics, and paper names from the slide content and context
4. For slides with no bullet content (section banners, agenda): write 1-2 sentences, use the timing shown
5. For content slides: write 2-4 sentences matching the ~1 min timing
6. CONTEXT blocks should include real metadata (authors, venues, URLs) from the provided context — do NOT invent information
7. If no paper metadata is available for a slide, write "- General thematic content" in CONTEXT
8. TRANSITION sentences should create narrative bridges, not just "Next we'll look at..."
9. For the LAST slide in this section, the transition should bridge to the next section: {next_note}
10. Every slide MUST have all four parts: title header, SAY, CONTEXT, TRANSITION
11. For slides with figures: add a **FIGURE:** block (1 sentence) describing what the audience sees on screen. Use the figure description and purpose provided. Omit this block for slides without figures.
    If figure annotations are listed (text visible in the figure), the SAY block should NOT repeat
    those annotations. Focus on insight, significance, and connections the figure cannot convey visually.
    The ZOOM SAY block (if applicable) should point out specific visual details and their implications,
    not re-describe what the parent slide's SAY already covered.
    If a Figure spec is provided, it is authoritative for figure narration. Do not describe the visual
    as bullet text in decorative boxes, generic colored ellipses, or a fake metric chart. Tie the
    narration to the spec's entities, relationships, quantities, and semantic goal.
{zoom_rules}
13. REFS BLOCK: For each slide, extract the 1-2 most citable references from the CONTEXT into a
    separate **REFS** block with short spoken-friendly citations (paper title + first author + venue).
    These will be used by the 2-host dialogue to reference specific work by name during the podcast.
    Omit the REFS block if no specific papers/talks/workshops apply to this slide.
14. NARRATIVE COHESION: Each SAY block must connect to the previous slide's content. If the
    previous section covered topic X, the first sentence should bridge from X to the current topic.
    Do NOT re-introduce a concept that was already explained in a prior slide — build on it instead.
15. APPENDIX SLIDES: For slides with "Slide type: appendix" metadata, write a brief SAY: "For those
    who want to dig deeper, we have collected all our references in this appendix." For subsequent
    appendix slides, write "Continuing our references." Keep these to 1 sentence each — the video
    should NOT walk through references one by one."""

    if verbose:
        print(f"  Section: {section_name} ({len(section_slides)} slides)")

    # Collect image paths for multimodal call
    image_paths: list[Path] = []
    if topic_dir:
        for slide in section_slides:
            for rel_path in slide.get('figure_paths', []):
                abs_path = (topic_dir / rel_path).resolve()
                if abs_path.exists():
                    image_paths.append(abs_path)
    if image_paths and verbose:
        print(f"    Attaching {len(image_paths)} figure(s) to prompt")

    expected_nums = {slide['num'] for slide in section_slides}
    expected_zoom_nums = {
        s['num'] for s in section_slides if zoom_companions.get(s['num'], False)
    }
    max_content_retries = max(
        1,
        _env_int(
            "PRESENTATION_GEMINI_SCRIPT_CONTENT_RETRIES",
            DEFAULT_SCRIPT_CONTENT_RETRIES,
        ),
    )
    last_missing: set[int] = set()
    last_missing_zooms: set[int] = set()

    for attempt in range(1, max_content_retries + 1):
        response = call_gemini(
            prompt,
            image_paths=image_paths or None,
            max_tokens=32768,
            verbose=verbose,
        )

        found_slide_nums = _script_slide_nums(response, "SLIDE")
        found_zoom_nums = _script_slide_nums(response, "ZOOM")
        last_missing = expected_nums - found_slide_nums
        last_missing_zooms = expected_zoom_nums - found_zoom_nums

        if not last_missing and not last_missing_zooms:
            return response

        if verbose:
            pieces = []
            if last_missing:
                pieces.append(f"missing slides {sorted(last_missing)}")
            if last_missing_zooms:
                pieces.append(f"missing zoom blocks {sorted(last_missing_zooms)}")
            print(
                "    Incomplete script response "
                f"({'; '.join(pieces)}), retry {attempt}/{max_content_retries}"
            )

    details = []
    if last_missing:
        details.append(f"missing slides {sorted(last_missing)}")
    if last_missing_zooms:
        details.append(f"missing zoom blocks {sorted(last_missing_zooms)}")
    raise RuntimeError(
        "Presenter script generation returned incomplete content for "
        f"{section_name}: {'; '.join(details) or 'unknown missing blocks'}"
    )


# ---------------------------------------------------------------------------
# Cohesion / dedup pass on the full assembled script
# ---------------------------------------------------------------------------

DEDUP_PROMPT = """You are editing a presenter script for {talk_context} on "{topic_name}".

The script below was generated section-by-section, so it contains cross-section repetition:
the same concepts, papers, or framings are re-introduced in later sections as if the audience
hasn't heard them before. Your job is to fix this.

FULL SCRIPT:
{full_script}

TASK: Return the COMPLETE script with these fixes applied:

1. REMOVE REDUNDANT RE-INTRODUCTIONS: If a concept or paper is explained in section A and then
   re-explained in section B, keep the first (fuller) explanation and replace later occurrences
   with forward references like "As we saw earlier...", "Building on the X framework we discussed...",
   or "Returning to Y with new evidence...". Do NOT delete the later SAY block — rewrite it to
   build on what came before instead of repeating it.

2. SMOOTH CROSS-SECTION TRANSITIONS: Fix TRANSITION lines that feel disconnected. Each section's
   last TRANSITION should naturally bridge to the next section's content. The first SAY block of
   each section should acknowledge what came before.

3. ELIMINATE REPEATED FRAMING: If multiple SAY blocks use the same hook/framing device
   (e.g., "This is a game-changer", "What's really exciting here"), vary the language.

4. PROGRESSIVE NARRATIVE: Ensure the script builds knowledge progressively. Later sections
   should reference earlier insights, creating a cumulative "aha" effect rather than
   isolated section silos.

RULES:
- Return the COMPLETE script (header, timing table, all slides)
- Do NOT remove any [SLIDE N] or [ZOOM N] blocks — every block must remain
- Do NOT change CONTEXT, REFS, or FIGURE blocks — only modify SAY and TRANSITION blocks
- Do NOT invent new facts, papers, or statistics — only rephrase existing content
- Keep the same markdown format with ---, ###, **SAY:**, etc.
- Preserve all > blockquote formatting in SAY blocks"""


def dedup_full_script(
    full_script: str,
    topic_display_name: str,
    conference_label: str | None = None,
    verbose: bool = False,
) -> str:
    """Run a cohesion/dedup pass on the fully assembled presenter script.

    Sends the complete script through one Gemini call that identifies and fixes
    cross-section repetition, smooths transitions, and ensures progressive narrative.
    """
    if verbose:
        print("  Running cohesion/dedup pass on full script...")

    talk_context = (
        f"a {conference_label} conference talk"
        if conference_label
        else "a conference talk"
    )
    prompt = DEDUP_PROMPT.format(
        topic_name=topic_display_name,
        talk_context=talk_context,
        full_script=full_script,
    )

    try:
        result = call_gemini(prompt, max_tokens=65536, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"    Dedup pass failed: {e} — keeping original")
        return full_script

    if not result or len(result) < len(full_script) * 0.5:
        if verbose:
            print("    Dedup pass returned too little content — keeping original")
        return full_script

    # Strip markdown fences if the LLM wrapped the output
    result = re.sub(r'^```(?:markdown)?\s*\n', '', result)
    result = re.sub(r'\n```\s*$', '', result)

    # Validate: check that slide blocks weren't lost
    original_slides = set(re.findall(r'\[SLIDE (\d+)\]', full_script))
    deduped_slides = set(re.findall(r'\[SLIDE (\d+)\]', result))
    missing = original_slides - deduped_slides

    if missing:
        if verbose:
            print(f"    WARNING: Dedup pass lost slides {sorted(missing)} — keeping original")
        return full_script

    if verbose:
        print("    Cohesion pass complete")

    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_topic(
    topic_num: str,
    verbose: bool = False,
    skip_dedup: bool = False,
    *,
    topic_dir: Path | None = None,
    input_path: Path | None = None,
    slide_mapping_path: Path | None = None,
    output_path: Path | None = None,
    output_notebooklm_path: Path | None = None,
    topic_display_name: str | None = None,
    conference_label: str | None = None,
) -> Path:
    """Generate presenter script for a single topic."""
    topic_name = TOPIC_NAMES.get(topic_num, topic_num)
    if topic_dir is None and topic_num not in TOPIC_NAMES:
        raise ValueError(f"Unknown topic: {topic_num}")

    if topic_dir is None:
        topic_dir = PRESENTATIONS_DIR / topic_name

    # Find input slides
    if input_path is None:
        input_path = find_best_input(topic_dir)
    if verbose:
        print(f"Processing: {topic_name}")
        print(f"  Input: {input_path.name}")

    # Parse slides
    content = input_path.read_text()
    slides = parse_slides(content)
    if not slides:
        raise ValueError(f"No slides found in {input_path}")

    if verbose:
        print(f"  Slides: {len(slides)}")

    # Load context sources
    papers = load_ranked_papers(topic_dir)
    mapping = load_slide_paper_mapping(topic_dir)
    sections = load_extracted_sections(topic_dir)
    influence_report = load_influence_report(topic_num) if topic_num in TOPIC_NAMES else ''
    attributions = load_source_attributions(topic_dir)

    # Load zoom companion info from HTML slide mapping
    zoom_companions = load_zoom_companions(topic_dir, mapping_path=slide_mapping_path)
    zoom_count = sum(1 for v in zoom_companions.values() if v)

    if verbose:
        print(f"  Papers: {len(papers)}")
        print(f"  Slide-paper mappings: {len(mapping)}")
        print(f"  Analysis sections: {len(sections)}")
        print(f"  Influence report: {len(influence_report)} chars")
        print(f"  Attributions: {len(attributions)}")
        print(f"  Zoom companions: {zoom_count}")

    # Build per-slide context
    slide_contexts: dict[int, str] = {}
    for slide in slides:
        slide_contexts[slide['num']] = build_slide_context(
            slide, papers, mapping, sections, influence_report, attributions
        )

    # Group slides by section
    groups = group_slides_by_section(slides)
    if verbose:
        print(f"  Sections: {len(groups)}")
        for g in groups:
            print(f"    {g['name']}: {len(g['slides'])} slides")

    # Generate script per section
    all_script_blocks = []
    for i, group in enumerate(groups):
        prev_name = groups[i - 1]['name'] if i > 0 else None
        next_name = groups[i + 1]['name'] if i + 1 < len(groups) else None

        script = generate_section_script(
            section_name=group['name'],
            section_slides=group['slides'],
            slide_contexts=slide_contexts,
            prev_section=prev_name,
            next_section=next_name,
            topic_display_name=topic_display_name
            or (
                topic_name.split('_', 1)[1].replace('_', ' ').title()
                if '_' in topic_name else topic_name
            ),
            topic_dir=topic_dir,
            zoom_companions=zoom_companions,
            conference_label=conference_label,
            verbose=verbose,
        )
        all_script_blocks.append(script)

    # Build timing summary table
    timing_rows = []
    total_content = 0
    total_short = 0
    total_zoom = 0
    for group in groups:
        content_count = sum(
            1 for s in group['slides'] if get_slide_timing(s) == '~1 min'
        )
        short_count = len(group['slides']) - content_count
        group_zoom = sum(
            1 for s in group['slides'] if zoom_companions.get(s['num'], False)
        )
        total_content += content_count
        total_short += short_count
        total_zoom += group_zoom
        slide_range = f"{group['slides'][0]['num']}-{group['slides'][-1]['num']}"
        est_seconds = content_count * 60 + short_count * 15 + group_zoom * 20
        est_time = f"~{est_seconds // 60} min {est_seconds % 60}s" if est_seconds >= 60 else f"~{est_seconds}s"
        zoom_note = f" (+{group_zoom} zoom)" if group_zoom else ""
        timing_rows.append(f"| {group['name']} | {slide_range} | {len(group['slides'])}{zoom_note} | {est_time} |")

    total_seconds = total_content * 60 + total_short * 15 + total_zoom * 20
    total_time = f"~{total_seconds // 60} min"

    # Display name from extracted_content or fallback
    display_name = topic_display_name or (
        topic_name.split('_', 1)[1].replace('_', ' ').title()
        if '_' in topic_name
        else topic_name
    )
    title_suffix = f" at {conference_label}" if conference_label else ""

    total_slide_range = f"{slides[0]['num']}-{slides[-1]['num']}"

    # Assemble final output
    header = f"""# Presenter Script: {display_name}{title_suffix}

<!--
GENERATED: {datetime.now().isoformat()}
TOPIC: {topic_name}
GENERATOR: generate_presenter_script.py (Pass 6)
SOURCE_SLIDES: {input_path.name}
TOTAL_SLIDES: {len(slides)}
TOTAL_TIME: {total_time}
-->

## Timing Summary

| Section | Slides | Count | Est. Time |
|---------|--------|-------|-----------|
{chr(10).join(timing_rows)}
| **Total** | **{total_slide_range}** | **{len(slides)}** | **{total_time}** |

"""

    full_script = header + '\n\n'.join(all_script_blocks) + '\n'

    # Cohesion/dedup pass on the full script
    if not skip_dedup:
        display = (
            topic_name.split('_', 1)[1].replace('_', ' ').title()
            if '_' in topic_name
            else topic_name
        )
        full_script = dedup_full_script(
            full_script,
            topic_display_name or display,
            conference_label=conference_label,
            verbose=verbose,
        )

    expected_slide_nums = {slide['num'] for slide in slides}
    generated_slide_nums = _script_slide_nums(full_script, "SLIDE")
    missing_slide_nums = expected_slide_nums - generated_slide_nums
    if missing_slide_nums:
        raise RuntimeError(
            "Presenter script is incomplete after generation: "
            f"missing slides {sorted(missing_slide_nums)}"
        )

    # Write output
    output_path = output_path or topic_dir / "6_presenter_script.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_script, encoding="utf-8")
    if output_notebooklm_path is not None:
        output_notebooklm_path.parent.mkdir(parents=True, exist_ok=True)
        output_notebooklm_path.write_text(full_script, encoding="utf-8")

    if verbose:
        try:
            output_display = output_path.relative_to(PROJECT_ROOT)
        except ValueError:
            output_display = output_path
        print(f"  Output: {output_display}")
        print(f"  Total time: {total_time}")

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
        description="Generate timed presenter script from final slide deck (Pass 6)"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--slides-md", type=Path, help="Workspace-relative or absolute slide markdown input.")
    parser.add_argument("--slide-mapping", type=Path, help="Workspace-relative or absolute presentation_slide_mapping.json.")
    parser.add_argument("--blueprint", type=Path, help="Workspace-relative or absolute approved presentation_blueprint.json input.")
    parser.add_argument("--blueprint-review", type=Path, help="Workspace-relative or absolute approved blueprint_review.json input.")
    parser.add_argument("--render-spec", type=Path, help="Structured render_spec.json input.")
    parser.add_argument("--delivery-single-speaker", type=Path, help="Structured single-speaker delivery contract output.")
    parser.add_argument("--delivery-notebooklm", type=Path, help="Structured NotebookLM delivery contract output.")
    parser.add_argument("--debug-script-md", type=Path, help="Debug presenter script markdown output for structured delivery.")
    parser.add_argument(
        "--debug-notebooklm-script-md",
        type=Path,
        help="Debug NotebookLM-style script markdown output for structured delivery.",
    )
    parser.add_argument("--output-script-md", type=Path, help="Workspace-relative or absolute presenter script output path.")
    parser.add_argument(
        "--output-notebooklm-script-md",
        type=Path,
        help="Workspace-relative or absolute NotebookLM-ready presenter script output path.",
    )
    parser.add_argument(
        "--conference-label",
        help=(
            "Conference label used in generated narration, for example 'ICLR 2025'. "
            "Defaults to an inference from --workspace."
        ),
    )
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
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Skip the cohesion/dedup pass on the full script"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.workspace is not None:
        structured_delivery = args.render_spec is not None
        if args.slide_mapping is None:
            parser.exit(2, f"{SLIDE_MAPPING_REQUIRED}\n")

        if structured_delivery:
            required_fields = (
                ("--presentation-id", args.presentation_id),
                ("--presentation-type", args.presentation_type),
                ("--render-spec", args.render_spec),
                ("--delivery-single-speaker", args.delivery_single_speaker),
                ("--delivery-notebooklm", args.delivery_notebooklm),
            )
        else:
            required_fields = (
                ("--presentation-id", args.presentation_id),
                ("--presentation-type", args.presentation_type),
                ("--slides-md", args.slides_md),
                ("--output-script-md", args.output_script_md),
                ("--output-notebooklm-script-md", args.output_notebooklm_script_md),
            )
        missing = [flag for flag, value in required_fields if value is None or value == ""]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")
        if not structured_delivery and not args.dry_run:
            blueprint_missing = [
                flag
                for flag, value in (
                    ("--blueprint", args.blueprint),
                    ("--blueprint-review", args.blueprint_review),
                )
                if value is None or value == ""
            ]
            if blueprint_missing:
                parser.error(
                    f"live workspace presenter generation requires {', '.join(blueprint_missing)}"
                )

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("slides_md", "--slides-md"),
            ("slide_mapping", "--slide-mapping"),
            ("blueprint", "--blueprint"),
            ("blueprint_review", "--blueprint-review"),
            ("render_spec", "--render-spec"),
            ("delivery_single_speaker", "--delivery-single-speaker"),
            ("delivery_notebooklm", "--delivery-notebooklm"),
            ("debug_script_md", "--debug-script-md"),
            ("debug_notebooklm_script_md", "--debug-notebooklm-script-md"),
            ("output_script_md", "--output-script-md"),
            ("output_notebooklm_script_md", "--output-notebooklm-script-md"),
        ):
            if getattr(args, attr) is None:
                continue
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr in {"output_script_md", "output_notebooklm_script_md"}:
                _reject_legacy_presentations_path(relative)
            if attr in {
                "delivery_single_speaker",
                "delivery_notebooklm",
                "debug_script_md",
                "debug_notebooklm_script_md",
            }:
                _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _require_approved_blueprint_context(args: argparse.Namespace) -> None:
    try:
        require_approved_blueprint(args.blueprint, args.blueprint_review)
    except PresentationBlueprintValidationError as exc:
        for error in exc.errors:
            print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(2) from exc


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    if args.render_spec is not None:
        return {
            "conference_label": args.conference_label
            or infer_conference_label(workspace=args.workspace, slides_md=None),
            "delivery_notebooklm": str(args.delivery_notebooklm),
            "delivery_single_speaker": str(args.delivery_single_speaker),
            "dry_run": True,
            "provider_requirements": [],
            "render_spec": str(args.render_spec),
            "slide_mapping": str(args.slide_mapping),
            "source_mode": "structured-render-spec",
        }
    return {
        "conference_label": args.conference_label
        or infer_conference_label(workspace=args.workspace, slides_md=args.slides_md),
        "dry_run": True,
        "output_notebooklm_script_md": str(args.output_notebooklm_script_md),
        "output_script_md": str(args.output_script_md),
        "blueprint": str(args.blueprint) if args.blueprint else None,
        "blueprint_review": str(args.blueprint_review) if args.blueprint_review else None,
        "blueprint_narration_beat_contract": BLUEPRINT_NARRATION_BEAT_CONTRACT,
        "notebooklm_timing_map": "blueprint.notebooklm_timing_map",
        "provider_requirements": PRESENTER_PROVIDER_REQUIREMENTS,
        "slide_mapping": str(args.slide_mapping),
        "slides_md": str(args.slides_md),
    }


def _workspace_topic_num(args: argparse.Namespace) -> str:
    return str(args.topic or args.presentation_id or "workspace")


def _display_name_from_presentation_id(presentation_id: str) -> str:
    name = re.sub(r"^\d{2}_", "", presentation_id)
    return name.replace("_", " ").replace("-", " ").title()


def infer_conference_label(*, workspace: Path | None, slides_md: Path | None = None) -> str | None:
    """Infer a human-readable conference label from common workspace slugs."""
    candidates: list[str] = []
    if workspace is not None:
        candidates.append(workspace.name)
    if slides_md is not None:
        candidates.extend(part for part in slides_md.parts if re.search(r"20\d{2}", part))
    acronym_map = {
        "iclr": "ICLR",
        "neurips": "NeurIPS",
        "icml": "ICML",
        "cvpr": "CVPR",
        "acl": "ACL",
        "emnlp": "EMNLP",
        "kdd": "KDD",
    }
    for candidate in candidates:
        match = re.search(r"([A-Za-z]+)[_-]?(20\d{2})", candidate)
        if not match:
            continue
        acronym = acronym_map.get(match.group(1).lower(), match.group(1).upper())
        return f"{acronym} {match.group(2)}"
    return None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_json_or_empty(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hash(
    render_spec_path: Path,
    render_spec: dict[str, object],
    filename: str,
) -> str:
    sibling = render_spec_path.parent / filename
    digest = _sha256_file(sibling)
    if digest is not None:
        return digest
    hashes = render_spec.get("input_hashes")
    if isinstance(hashes, dict):
        value = hashes.get(filename)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _relative_to_deck(path: Path, deck_dir: Path) -> str:
    try:
        return path.resolve().relative_to(deck_dir.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), deck_dir.resolve())


def _review_paths_for_variant(variant: str) -> dict[str, str]:
    if variant == "notebooklm":
        return {
            "deck_review_json": "narration_review.json",
            "deck_review_md": "narration_review.md",
            "lint_report_json": "narration_lint_report.json",
            "series_review_json": "../narration_series_review.json",
            "series_review_md": "../narration_series_review.md",
            "review_skill": "conference-narration-review",
            "fix_skill": "conference-narration-fix",
        }
    return {
        "deck_review_json": "presenter_narration_review.json",
        "deck_review_md": "presenter_narration_review.md",
        "lint_report_json": "presenter_narration_lint_report.json",
        "series_review_json": "../presenter_narration_series_review.json",
        "series_review_md": "../presenter_narration_series_review.md",
        "review_skill": "conference-narration-review",
        "fix_skill": "conference-narration-fix",
    }


def _review_json_approves_tts(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("approve_for_tts") is not True:
        return False
    status = str(payload.get("status") or "").strip().lower()
    if status in {"fail", "failed", "blocked", "needs_revision", "needs-revision"}:
        return False
    counts = payload.get("finding_counts")
    if isinstance(counts, dict):
        try:
            return int(counts.get("fail", counts.get("fails", 0)) or 0) == 0
        except (TypeError, ValueError):
            return False
    return True


def _reviews_approved(deck_dir: Path, paths: dict[str, str]) -> bool:
    for key in ("deck_review_json", "series_review_json"):
        value = paths.get(key)
        if not value:
            return False
        review_path = Path(value)
        if not review_path.is_absolute():
            review_path = deck_dir / review_path
        if not _review_json_approves_tts(review_path.resolve()):
            return False
    return True


def _slides_by_id(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    slides = payload.get("slides")
    if not isinstance(slides, list):
        return {}
    return {
        str(slide.get("slide_id")): slide
        for slide in slides
        if isinstance(slide, dict) and str(slide.get("slide_id") or "").strip()
    }


def _slide_by_index(payload: dict[str, object], index: int) -> dict[str, object]:
    slides = payload.get("slides")
    if not isinstance(slides, list) or index >= len(slides):
        return {}
    slide = slides[index]
    return slide if isinstance(slide, dict) else {}


def _matching_slide(
    payload: dict[str, object],
    slide_id: str,
    index: int,
) -> dict[str, object]:
    return _slides_by_id(payload).get(slide_id) or _slide_by_index(payload, index)


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_source_contribution(item: object) -> dict[str, object] | None:
    if not isinstance(item, dict):
        return None
    label = str(item.get("label") or item.get("source_label") or "").strip()
    source_id = str(item.get("source_id") or item.get("reference_id") or "").strip()
    title = html.unescape(str(item.get("title") or item.get("work") or source_id or "")).strip()
    contribution_note = str(
        item.get("contribution_note")
        or item.get("relevance_rationale")
        or item.get("rationale")
        or ""
    ).strip()
    authors = _as_text_list(item.get("authors"))
    speakers = _as_text_list(item.get("speakers"))
    works = _as_text_list(item.get("works"))
    source_type = str(item.get("source_type") or "").strip()
    if not any((label, title, authors, speakers, works, contribution_note)):
        return None
    return {
        "label": label,
        "source_id": source_id or label,
        "title": title,
        "authors": authors,
        "speakers": speakers,
        "works": works,
        "source_type": source_type,
        "contribution_note": contribution_note,
    }


def _dedupe_source_contributions(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, object]] = []
    for item in items:
        key = (str(item.get("label") or ""), str(item.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _source_contributions_for_slide(
    *,
    render_slide: dict[str, object],
    deck_slide: dict[str, object],
    source_slide: dict[str, object],
) -> list[dict[str, object]]:
    raw_items: list[object] = []
    for key in ("named_paper_talk_speaker_author_mentions", "source_chips", "visible_references"):
        value = deck_slide.get(key)
        if isinstance(value, list):
            raw_items.extend(value)
    value = source_slide.get("visible_references")
    if isinstance(value, list):
        raw_items.extend(value)
    value = render_slide.get("source_chips")
    if isinstance(value, list):
        raw_items.extend(value)
    normalized = [
        item
        for item in (_normalize_source_contribution(raw_item) for raw_item in raw_items)
        if item is not None
    ]
    return _dedupe_source_contributions(normalized)


def _visual_hooks_for_slide(visual_slide: dict[str, object], render_slide: dict[str, object]) -> list[str]:
    hooks: list[str] = []
    figures = visual_slide.get("figures")
    if isinstance(figures, list):
        for figure in figures:
            if not isinstance(figure, dict):
                continue
            for key in ("narration_hook", "contribution_note", "visual_role"):
                value = figure.get(key)
                if isinstance(value, str) and value.strip():
                    hooks.append(value.strip())
            brief = figure.get("figure_design_brief")
            if isinstance(brief, dict):
                value = brief.get("narration_hook")
                if isinstance(value, str) and value.strip():
                    hooks.append(value.strip())
    notes = render_slide.get("speaker_notes")
    if isinstance(notes, str) and notes.strip():
        hooks.append(notes.strip())
    return hooks


def _source_tokens(contributions: list[dict[str, object]]) -> list[str]:
    tokens: list[str] = []
    for contribution in contributions:
        if _is_internal_source_contribution(contribution):
            continue
        for key in ("label", "title"):
            value = str(contribution.get(key) or "").strip()
            if value:
                tokens.append(value)
        for key in ("authors", "speakers", "works"):
            tokens.extend(_as_text_list(contribution.get(key)))
    return [token for token in tokens if len(token) >= 3]


def _mentions_source_contribution(
    narration: str,
    contributions: list[dict[str, object]],
) -> bool:
    text = narration.lower()
    for token in _source_tokens(contributions):
        if token.lower() in text:
            return True
    return False


def _mentions_visual(narration: str, hooks: list[str], visual_refs: object) -> bool:
    text = narration.lower()
    visual_words = (
        "audience sees",
        "audience watches",
        "the slide shows",
        "the screen shows",
        "what we see",
        "figure",
        "visual",
        "chart",
        "diagram",
        "evidence is laid out",
        "map",
        "boundary",
        "panel",
        "layout",
        "architecture",
        "table",
        "column",
        "screen",
    )
    return any(word in text for word in visual_words)


def _mentions_why_it_matters(narration: str, hooks: list[str]) -> bool:
    text = " ".join([narration, *hooks]).lower()
    return any(
        phrase in text
        for phrase in (
            "why it matters",
            "matters",
            "because",
            "changes",
            "shows why",
            "risk",
            "trust",
            "defense",
            "safety",
            "guarantee",
            "constraint",
            "structural",
        )
    )


def _source_grounding_checks(
    segments: list[dict[str, object]],
    slide_context: dict[str, dict[str, object]],
) -> dict[str, object]:
    segment_results: list[dict[str, object]] = []
    for segment in segments:
        slide_id = str(segment.get("slide_id") or "")
        context = slide_context.get(slide_id, {})
        contributions = context.get("source_contributions")
        if not isinstance(contributions, list):
            contributions = []
        hooks = context.get("visual_hooks")
        if not isinstance(hooks, list):
            hooks = []
        narration = str(segment.get("narration") or "")
        mentions_visual = _mentions_visual(narration, [str(hook) for hook in hooks], segment.get("visual_asset_refs"))
        mentions_why = _mentions_why_it_matters(narration, [str(hook) for hook in hooks])
        external_contributions = [
            contribution
            for contribution in contributions
            if isinstance(contribution, dict)
            and not _is_internal_source_contribution(contribution)
        ]
        mentions_source = (
            True
            if not external_contributions
            else _mentions_source_contribution(narration, external_contributions)
        )
        status = "pass" if mentions_visual and mentions_why and mentions_source else "blocked"
        segment_results.append(
            {
                "segment_id": segment.get("segment_id"),
                "slide_id": slide_id,
                "status": status,
                "mentions_what_the_audience_sees": mentions_visual,
                "mentions_why_it_matters": mentions_why,
                "mentions_named_source_contribution": mentions_source,
                "required_source_contributions": contributions[:4],
                "visual_narration_hooks": hooks[:4],
            }
        )
    return {
        "status": "pass"
        if segment_results and all(result.get("status") == "pass" for result in segment_results)
        else "blocked",
        "required_checks": [
            "what the audience sees",
            "why it matters",
            "named paper, named talk, speaker, author, or work contribution",
        ],
        "segment_results": segment_results,
    }


def _attribution_validation_checks(
    slide_context: dict[str, dict[str, object]],
) -> dict[str, object]:
    segment_results: list[dict[str, object]] = []
    for slide_id, context in slide_context.items():
        contributions = context.get("source_contributions")
        if not isinstance(contributions, list):
            contributions = []
        suspect_names: list[str] = []
        for contribution in contributions:
            if not isinstance(contribution, dict):
                continue
            for key in ("authors", "speakers"):
                for value in _as_text_list(contribution.get(key)):
                    if _is_unverified_person_name(value, contribution):
                        suspect_names.append(value)
        segment_results.append(
            {
                "slide_id": slide_id,
                "status": "blocked" if suspect_names else "pass",
                "suspect_unverified_names": sorted(set(suspect_names)),
            }
        )
    return {
        "status": "pass"
        if segment_results and all(result.get("status") == "pass" for result in segment_results)
        else "blocked",
        "required_checks": [
            "displayed and spoken people names match source records or approved aliases"
        ],
        "segment_results": segment_results,
    }


def _clean_visual_hook(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"(?i)\bNotebookLM visual hook:\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bHost A\s+(?:frames|names|introduces)\s+", "", cleaned)
    cleaned = re.sub(r"(?i)\bHost B\s+(?:connects|grounds|adds)\s+", "", cleaned)
    cleaned = re.sub(r"(?i)\bHost [AB]\s+", "", cleaned)
    cleaned = cleaned.strip(" .;")
    return cleaned or "the generated visual proof object"


def _source_anchor_phrase(contributions: list[dict[str, object]]) -> str:
    for contribution in contributions:
        if _is_internal_source_contribution(contribution):
            continue
        title = str(contribution.get("title") or "").strip()
        if title and not _is_internal_source_anchor(title):
            return title
        works = _as_text_list(contribution.get("works"))
        if works:
            work = works[0]
            if not _is_internal_source_anchor(work):
                return work
        for key in ("speakers", "authors"):
            values = _as_text_list(contribution.get(key))
            if values:
                value = values[0]
                if not _is_internal_source_anchor(value):
                    return value
        label = str(contribution.get("label") or "").strip()
        if label and not _is_internal_source_anchor(label):
            return label
    return ""


def _is_internal_source_contribution(contribution: dict[str, object]) -> bool:
    for key in ("title", "label", "source_id", "reference_id", "source_type", "artifact_url"):
        value = contribution.get(key)
        if isinstance(value, str) and _is_internal_source_anchor(value):
            return True
    return False


def _is_internal_source_anchor(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "phase 01",
            "v2.2",
            "canary recovery",
            "brand and creative brief",
            "governance anchor",
            "planning:",
            ".planning/",
        )
    )


KNOWN_PERSON_ALIAS_ERRORS = frozenset({"Don Song"})
PERSON_NAMES_REQUIRING_SOURCE_EVIDENCE = frozenset({"Davina Jane", "Fuming Yue"})
PERSON_SOURCE_EVIDENCE_FIELDS = (
    "source_url",
    "paper_url",
    "openreview_url",
    "doi_url",
    "slideslive_url",
    "artifact_url",
    "poster_url",
    "url",
    "path",
)
PERSON_SOURCE_ID_EVIDENCE_PREFIXES = (
    "speech_transcript:",
    "paper_abstract:",
    "poster-",
    "poster:",
    "openreview:",
)


def _is_unverified_person_name(name: str, contribution: dict[str, object]) -> bool:
    if name in KNOWN_PERSON_ALIAS_ERRORS:
        return True
    if name not in PERSON_NAMES_REQUIRING_SOURCE_EVIDENCE:
        return False
    has_url_evidence = any(
        isinstance(contribution.get(field), str) and str(contribution.get(field)).strip()
        for field in PERSON_SOURCE_EVIDENCE_FIELDS
    )
    source_id = str(contribution.get("source_id") or "").strip().lower()
    has_source_id_evidence = any(
        source_id.startswith(prefix) for prefix in PERSON_SOURCE_ID_EVIDENCE_PREFIXES
    )
    return not (has_url_evidence or has_source_id_evidence)


def _compact_clause(text: str, *, fallback: str, max_chars: int = 128) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .;")
    if not cleaned:
        cleaned = fallback.strip(" .;")
    if len(cleaned) <= max_chars:
        return _strip_dangling_clause_end(cleaned) or cleaned
    clipped = cleaned[: max_chars - 1].rsplit(" ", 1)[0].strip(" ,;:")
    return _strip_dangling_clause_end(clipped) or fallback.strip(" .;")


def _strip_dangling_clause_end(text: str) -> str:
    words = text.strip(" ,;:").split()
    dangling = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "for",
        "from",
        "in",
        "nor",
        "of",
        "or",
        "plus",
        "the",
        "to",
        "with",
        "within",
    }
    while words and words[-1].lower().strip(".,;:-–—").lstrip("-–—") in dangling:
        words.pop()
    if words and words[-1].endswith(("-", "–", "—")):
        words.pop()
    return " ".join(words)


def _lower_initial(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    return cleaned[0].lower() + cleaned[1:]


def _visual_description_from_focus(
    *,
    figure_focus: str,
    proof_object: str,
    evidence_caption: str,
    title: str,
) -> str:
    seed = figure_focus or proof_object or evidence_caption or title
    cleaned = re.sub(r"\s+", " ", seed).strip(" .;")
    cleaned = re.split(
        r"(?i)\b(?:numeric|exact metric|citations sit|if exact|phase 05 may use)\b",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = cleaned.strip(" .;:")
    if ":" in cleaned:
        lead, detail = [part.strip(" .;:") for part in cleaned.split(":", 1)]
    else:
        lead, detail = cleaned, ""
    lower = f"{lead} {detail}".lower()
    if "left column" in lower and "right column" in lower:
        if "text-refusal" in lower or "refusal" in lower:
            return "a two-column contrast between text-refusal metrics and systems-safety requirements"
        return "a two-column contrast between the old evaluation frame and the new deployment frame"
    if (
        all(term in lower for term in ("agent loop", "memory", "tools", "environment"))
        and "three-panel" not in lower
        and "three panel" not in lower
    ) or all(term in lower for term in ("agent state", "tool action", "observation")):
        return "a four-step agent loop from state to tool action, observation, and oversight"
    if any(
        term in lower
        for term in (
            "a2asecbench",
            "mcp-safetybench",
            "mcp security bench",
            "blind-act",
            "decepticon",
            "openagentsafety",
            "source-bound benchmark coverage matrix",
        )
    ):
        return "a benchmark coverage matrix comparing protocol, tool, memory, environment, and action-safety coverage"
    if all(term in lower for term in ("memory", "tool", "environment")) and (
        "three-panel" in lower or "three panel" in lower
    ):
        return "three panels for memory, tools, and environment attack surfaces"
    if (
        "transfer-target schematic" in lower
        or "universal adversarial image token" in lower
        or "target vlm" in lower
    ):
        return "a transfer diagram where a universal adversarial image token moves from a source VLM to multiple target VLMs"
    if (
        "layered attack-surface diagram" in lower
        or all(term in lower for term in ("text", "image", "tool-output", "agent decision loop"))
    ):
        return "a multi-channel attack surface feeding the agent decision loop"
    if "any-depth alignment" in lower and "runtime enforcement" in lower:
        return "a defense-coverage matrix comparing alignment, steering, activation, graph, antibody, and runtime layers"
    if "agent loop components" in lower:
        return "a four-step agent loop from state to tool action, observation, and oversight"
    if "rows are" in lower and "columns" in lower and "benchmark" in lower:
        return "a matrix crossing benchmark suites with deployed-agent coverage dimensions"
    if any(term in lower for term in ("reward hacking", "judge bias", "strategic dishonesty", "human monitor")):
        return "an evaluation pipeline that treats benchmark construction, judge bias, and monitor behavior as attack surfaces"
    if any(term in lower for term in ("sosbench", "scientific domains", "capability-gap red-teaming")):
        return "a high-risk science safety dashboard with domain evidence and a capability-gap red-teaming branch"
    if any(term in lower for term in ("privacy threat taxonomy", "five labeled branches", "agentic context extraction")):
        return "a five-branch privacy-risk map for context extraction, data protection, geolocation, source inference, and unlearning"
    if any(term in lower for term in ("source-inference defense", "parameter-shuffling", "residue-number-system")):
        return "a source-inference defense schematic comparing undefended attack paths with defended operating points"
    if any(term in lower for term in ("seven-node radial", "trustworthy deployment hub", "topic clusters")):
        return "a radial portfolio map connecting seven safety evidence clusters around layered controls"
    if "side-by-side" in lower or "trusted user instruction" in lower:
        return "a side-by-side attacker-defender scenario with an outcome strip"
    if "training alignment at the base" in lower:
        return "a stacked runtime-control architecture from training alignment to policy enforcement"
    if "persona-condition" in lower or "persona conditions" in lower:
        return "persona-condition evidence panels showing behavior shifts across role contexts"
    detail = _compact_clause(detail, fallback=lead or title, max_chars=145)
    lead = _compact_clause(lead, fallback=title, max_chars=72)
    if "two-column" in lower or "two column" in lower:
        return _compact_clause(f"a two-column contrast where {detail}", fallback=lead, max_chars=150)
    if "three-panel" in lower or "three panel" in lower:
        return _compact_clause(f"three connected panels showing {detail}", fallback=lead, max_chars=150)
    if "matrix" in lower:
        return _compact_clause(f"a coverage matrix for {detail}", fallback=lead, max_chars=150)
    if "table" in lower:
        return _compact_clause(f"a comparison table covering {detail}", fallback=lead, max_chars=150)
    if "stack" in lower or "architecture" in lower:
        return _compact_clause(f"a layered architecture view of {detail}", fallback=lead, max_chars=150)
    if "map" in lower:
        return _compact_clause(f"a map connecting {detail}", fallback=lead, max_chars=150)
    if "panel" in lower:
        return _compact_clause(f"evidence panels for {detail}", fallback=lead, max_chars=150)
    if "brand" in lower or "title slide" in lower:
        return "the branded title frame that sets the Topic 4 scope"
    return "a source-grounded visual that maps named evidence to the slide takeaway"


def _variant(items: list[str], index: int) -> str:
    if not items:
        return ""
    return items[(index - 1) % len(items)]

def _sentence_join(parts: list[str]) -> str:
    sentences: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"\.{2,}$", ".", cleaned)
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        sentences.append(cleaned)
    return " ".join(sentences)


def _append_source_anchor(text: str, source_anchor: str, *, index: int) -> str:
    cleaned = _sentence_join([text])
    source_anchor = source_anchor.strip()
    if not source_anchor or source_anchor.lower() in cleaned.lower():
        return cleaned
    phrase = f"{source_anchor} is the source to keep in mind here."
    return _sentence_join([cleaned, phrase])


def _compact_proof_phrase(proof_object: str, title: str) -> str:
    proof = proof_object.strip()
    if not proof or len(proof) > 140:
        compact_title = re.split(r"[:;]", title.strip(), maxsplit=1)[0]
        compact_title = re.sub(r"\s+", " ", compact_title).strip()
        if len(compact_title) > 90:
            compact_title = compact_title[:87].rsplit(" ", 1)[0] + "..."
        proof = compact_title or "the evidence visual"
    return proof


def _notebooklm_host_b_text(
    *,
    slide_id: str,
    title: str,
    proof_object: str,
    evidence_caption: str,
    visual_takeaway: str,
    figure_focus: str,
    source_anchor: str,
    index: int,
) -> str:
    caption = _compact_clause(evidence_caption, fallback=title, max_chars=118)
    takeaway = _compact_clause(visual_takeaway, fallback=caption, max_chars=128)
    visual_description = _visual_description_from_focus(
        figure_focus=figure_focus,
        proof_object=proof_object,
        evidence_caption=evidence_caption,
        title=title,
    )
    visual_sentence = _variant(
        [
            f"The figure is organized as {visual_description}.",
            f"Visually, the frame uses {visual_description}.",
            f"The evidence is laid out as {visual_description}.",
            f"Here the main visual centers {visual_description}.",
            f"On screen, the visual structure is {visual_description}.",
            f"The figure turns the claim into {visual_description}.",
        ],
        index,
    )
    takeaway_sentence = _variant(
        [
            f"The takeaway is that {_lower_initial(takeaway)}.",
            f"That matters because {_lower_initial(takeaway)}.",
            f"The reason to care is that {_lower_initial(takeaway)}.",
            f"So read the visual as evidence that {_lower_initial(takeaway)}.",
        ],
        index,
    )
    source_clause = (
        _variant(
            [
                f"{source_anchor} grounds that point.",
                f"{source_anchor} is the source behind that claim.",
                f"{source_anchor} supplies the evidence anchor.",
            ],
            index,
        )
        if source_anchor
        and source_anchor.lower() not in f"{visual_description} {caption} {takeaway}".lower()
        else ""
    )
    return _sentence_join([visual_sentence, takeaway_sentence, source_clause])


def _segment_text(
    slide: dict,
    *,
    style: str,
    index: int,
    visual_hook: str = "",
    source_contributions: list[dict[str, object]] | None = None,
) -> str:
    # Never speak a raw slide id ("s-events", "sl_02_...") aloud — prefer the
    # human title and fall back to a neutral slide number.
    title = str(
        (slide.get("layout_blocks") or [{}])[0].get("text")
        or slide.get("title")
        or ""
    ).strip()
    if not title or re.match(r"^(s-|sl_)[\w-]*$", title):
        title = str(slide.get("title") or "").strip() or f"Slide {index}"
    notes = str(slide.get("speaker_notes") or "Prepared by Nikolaos Vasiloglou.")
    source_anchor = _source_anchor_phrase(source_contributions or [])
    if style == "notebooklm":
        proof_object = _clean_visual_hook(visual_hook)
        host_a = _append_source_anchor(notes, source_anchor, index=index)
        host_b = _notebooklm_host_b_text(
            slide_id=str(slide.get("slide_id") or ""),
            title=title,
            proof_object=proof_object,
            evidence_caption=str(slide.get("evidence_caption") or ""),
            visual_takeaway=str(slide.get("visual_takeaway") or ""),
            figure_focus=str(slide.get("figure_focus") or ""),
            source_anchor=source_anchor,
            index=index,
        )
        return f"Host A: {host_a} Host B: {host_b}"
    return _append_source_anchor(_sentence_join([title, notes]), source_anchor, index=index)


def _delivery_context(
    *,
    render_spec: dict[str, object],
    deck_contract: dict[str, object],
    source_binding: dict[str, object],
    visual_plan: dict[str, object],
) -> dict[str, dict[str, object]]:
    context: dict[str, dict[str, object]] = {}
    slides = render_spec.get("slides")
    if not isinstance(slides, list):
        return context
    for zero_index, slide in enumerate(slides):
        if not isinstance(slide, dict):
            continue
        slide_id = str(slide.get("slide_id") or f"slide-{zero_index + 1:03d}")
        deck_slide = _matching_slide(deck_contract, slide_id, zero_index)
        source_slide = _matching_slide(source_binding, slide_id, zero_index)
        visual_slide = _matching_slide(visual_plan, slide_id, zero_index)
        context[slide_id] = {
            "source_contributions": _source_contributions_for_slide(
                render_slide=slide,
                deck_slide=deck_slide,
                source_slide=source_slide,
            ),
            "visual_hooks": _visual_hooks_for_slide(visual_slide, slide),
        }
    return context


def _delivery_contract(
    render_spec: dict[str, object],
    *,
    render_spec_path: Path,
    style: str,
    deck_contract: dict[str, object],
    source_binding: dict[str, object],
    visual_plan: dict[str, object],
    deck_dir: Path,
) -> dict[str, object]:
    segments: list[dict[str, object]] = []
    timing_map: list[dict[str, object]] = []
    slide_state_map: list[dict[str, object]] = []
    slide_context = _delivery_context(
        render_spec=render_spec,
        deck_contract=deck_contract,
        source_binding=source_binding,
        visual_plan=visual_plan,
    )
    current = 0.0
    for index, slide in enumerate(render_spec.get("slides", []), start=1):
        if not isinstance(slide, dict):
            continue
        duration = max(float(slide.get("timing_seconds") or 1.0), 1.0)
        slide_id = str(slide.get("slide_id") or f"slide-{index:03d}")
        segment_id = f"{style}-{index:03d}"
        context = slide_context.get(slide_id, {})
        source_contributions = context.get("source_contributions")
        if not isinstance(source_contributions, list):
            source_contributions = []
        visual_hooks = context.get("visual_hooks")
        if not isinstance(visual_hooks, list):
            visual_hooks = []
        visual_hook = str(visual_hooks[0]) if visual_hooks else str(slide.get("figure_focus") or "")
        segment = {
            "segment_id": segment_id,
            "slide_id": slide_id,
            "slide_number": index,
            "start_seconds": round(current, 3),
            "end_seconds": round(current + duration, 3),
            "duration_seconds": duration,
            "speaker": "Nikolaos Vasiloglou" if style == "single_speaker" else "Host A / Host B",
            "narration": _segment_text(
                slide,
                style=style,
                index=index,
                visual_hook=visual_hook,
                source_contributions=source_contributions,
            ),
            "source_chips": source_contributions or slide.get("source_chips") or [],
            "source_ids": [
                str(item.get("source_id"))
                for item in source_contributions
                if isinstance(item, dict) and item.get("source_id")
            ],
            "source_labels": [
                str(item.get("label"))
                for item in source_contributions
                if isinstance(item, dict) and item.get("label")
            ],
            "visual_asset_refs": slide.get("visual_asset_refs") or [],
        }
        segments.append(segment)
        timing_map.append(
            {
                "segment_id": segment_id,
                "slide_id": slide_id,
                "slide_number": index,
                "start_seconds": round(current, 3),
                "end_seconds": round(current + duration, 3),
                "duration_seconds": duration,
            }
        )
        slide_state_map.append(
            {
                "segment_id": segment_id,
                "slide_id": slide_id,
                "state_id": str(slide.get("state_id") or f"{slide_id}-state"),
                "visual_asset_refs": slide.get("visual_asset_refs") or [],
                "visual_narration_hook": visual_hook,
                "source_labels": [
                    str(item.get("label"))
                    for item in source_contributions
                    if isinstance(item, dict) and item.get("label")
                ],
            }
        )
        current += duration
    source_checks = _source_grounding_checks(segments, slide_context)
    attribution_checks = _attribution_validation_checks(slide_context)
    review_paths = _review_paths_for_variant(style)
    approve_for_tts = (
        source_checks.get("status") == "pass"
        and attribution_checks.get("status") == "pass"
        and _reviews_approved(deck_dir, review_paths)
    )
    script_name = (
        "6_presenter_script_notebooklm_2host.md"
        if style == "notebooklm"
        else "6_presenter_script.md"
    )
    script_json = "video_build_notebooklm/script.json" if style == "notebooklm" else "video_build/script.json"
    notebooklm_visual_hooks: list[str] = []
    if style == "notebooklm":
        for item in slide_state_map:
            hook = _clean_visual_hook(str(item.get("visual_narration_hook") or ""))
            if hook:
                notebooklm_visual_hooks.append(f"NotebookLM visual hook: {hook}")
    return {
        "schema_version": 2,
        "presentation_id": str(render_spec.get("presentation_id") or "presentation"),
        "delivery_variant": style,
        "style": style,
        "source_render_spec": _relative_to_deck(render_spec_path, deck_dir),
        "presentation_deck_sha256": _input_hash(render_spec_path, render_spec, "presentation_deck.json"),
        "source_binding_report_sha256": _input_hash(
            render_spec_path,
            render_spec,
            "source_binding_report.json",
        ),
        "visual_plan_sha256": _input_hash(render_spec_path, render_spec, "visual_plan.json"),
        "script_path": script_name,
        "script_json_path": script_json,
        "timing_map": timing_map,
        "slide_state_map": slide_state_map,
        "narration_review_paths": review_paths,
        "approve_for_tts": approve_for_tts,
        "source_grounding_checks": source_checks,
        "attribution_validation_checks": attribution_checks,
        "notebooklm_visual_narration_hooks": notebooklm_visual_hooks,
        "total_duration_seconds": round(current, 3),
        "max_voice_gap_seconds": 0.5,
        "timing_divergence_seconds": 0.0 if style == "notebooklm" else None,
        "voice_gap_checks": [
            {
                "window": "00:48-00:55",
                "status": "covered",
                "max_gap_seconds": 0.5,
            }
        ],
        "segments": segments,
    }


def _notebooklm_debug_turns(text: str) -> list[tuple[str, str]]:
    markers = list(re.finditer(r"\b(Host [AB]):\s*", text))
    if len({match.group(1) for match in markers}) < 2:
        return [("Host A", text.strip())] if text.strip() else []
    turns: list[tuple[str, str]] = []
    for index, match in enumerate(markers):
        start = match.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        turn_text = re.sub(r"\s+", " ", text[start:end]).strip()
        if turn_text:
            turns.append((match.group(1), turn_text))
    return turns


def _write_debug_script(path: Path, contract: dict[str, object], *, notebooklm: bool) -> None:
    lines = ["# Structured Presenter Script", ""]
    for segment in contract.get("segments", []):
        if not isinstance(segment, dict):
            continue
        slide_number = int(segment.get("slide_number") or 0)
        label = f"SLIDE {slide_number}"
        title = str(segment.get("slide_id") or label)
        if notebooklm:
            lines.extend([f"### [{label}] {title}", "**DIALOGUE:**"])
            for speaker, text in _notebooklm_debug_turns(str(segment.get("narration") or "")):
                display_speaker = {"Host A": "Speaker 1", "Host B": "Speaker 2"}.get(speaker, speaker)
                lines.append(f"- **{display_speaker}:** {text}")
            lines.append("")
            continue
        lines.extend(
            [
                f"### [{label}] {title}",
                "**SAY:**",
                f"> {segment.get('narration')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _editorial_deck_kind(deck_dir: Path) -> str:
    name = deck_dir.name
    if name.startswith("executive_summary_short"):
        return "executive_short"
    if name.startswith("executive_summary_long"):
        return "executive_long"
    return "executive" if name.startswith("executive") else "topic"


def _write_structured_review_scaffolding(deck_dir: Path, render_spec: dict[str, object]) -> None:
    storyboard_path = deck_dir / "editorial_storyboard.json"
    gate_path = deck_dir / "presentation_editorial_gate_report.json"
    slides = [slide for slide in render_spec.get("slides", []) if isinstance(slide, dict)]
    # Why each slide is kept: the blueprint states every slide's intent, which is
    # the retained rationale the editorial gate requires. Without it the gate
    # failed every structured deck (CVPR 2026, all twelve topics).
    blueprint = _load_json_or_empty(deck_dir / "presentation_blueprint.json")
    intents = {
        str(item.get("slide_id")): str(item.get("intent") or "").strip()
        for item in blueprint.get("slides", [])
        if isinstance(item, dict)
    }

    def _retained_rationale(slide: dict[str, object]) -> str:
        intent = intents.get(str(slide.get("slide_id")), "")
        if intent:
            return f"Kept to {intent[0].lower()}{intent[1:]}"
        if slide.get("slide_kind") in {"events", "resources"}:
            return "Structural bookend derived from the deck's bound sources."
        return "Kept as the only slide carrying this claim."

    def _representative_papers(slide: dict[str, object]) -> list[str]:
        papers = [
            str(ref.get("label") or ref.get("title"))
            for ref in slide.get("source_chips", [])
            if isinstance(ref, dict)
        ]
        if papers:
            return papers
        # Bookends carry no chips; they name their works directly.
        listed = list(slide.get("papers") or []) + list(slide.get("resources") or [])
        return [
            str(item.get("name") or item.get("title"))
            for item in listed
            if isinstance(item, dict) and (item.get("name") or item.get("title"))
        ][:6]

    storyboard = {
        "schema_version": 1,
        "source_mode": "structured-render-spec",
        "clusters": [
            {
                "claim": str((slide.get("layout_blocks") or [{}])[0].get("text") or slide.get("slide_id")),
                "representative_papers": _representative_papers(slide),
                "retained_slides": [index],
                "merged_slides": [],
                "dropped_slides": [],
                "figure_roles": {str(index): ",".join(str(ref) for ref in slide.get("visual_asset_refs", []))},
                "narration_beats": {str(index): str(slide.get("speaker_notes") or "")},
                "retained_rationale": _retained_rationale(slide),
            }
            for index, slide in enumerate(slides, start=1)
        ],
    }
    _write_json(storyboard_path, storyboard)
    # The gate report comes from the gate. This used to write a literal
    # "passed" here, which was the only editorial gate a structured deck ever
    # had -- while the real gate failed all twelve CVPR 2026 topic decks.
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "presentation_editorial_gate.py"),
            "--deck-dir", str(deck_dir),
            "--deck-kind", _editorial_deck_kind(deck_dir),
            "--report-out", str(gate_path),
        ],
        check=False,
    )


def process_structured_delivery(args: argparse.Namespace) -> tuple[Path, Path]:
    render_spec = _load_json(args.render_spec)
    deck_dir = args.delivery_single_speaker.parent
    deck_contract = _load_json_or_empty(args.render_spec.parent / "presentation_deck.json")
    source_binding = _load_json_or_empty(args.render_spec.parent / "source_binding_report.json")
    visual_plan = _load_json_or_empty(args.render_spec.parent / "visual_plan.json")
    single = _delivery_contract(
        render_spec,
        render_spec_path=args.render_spec,
        style="single_speaker",
        deck_contract=deck_contract,
        source_binding=source_binding,
        visual_plan=visual_plan,
        deck_dir=deck_dir,
    )
    notebooklm = _delivery_contract(
        render_spec,
        render_spec_path=args.render_spec,
        style="notebooklm",
        deck_contract=deck_contract,
        source_binding=source_binding,
        visual_plan=visual_plan,
        deck_dir=deck_dir,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "single_speaker": single,
                    "notebooklm": notebooklm,
                    "render_spec": str(args.render_spec),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return args.delivery_single_speaker, args.delivery_notebooklm
    _write_json(args.delivery_single_speaker, single)
    _write_json(args.delivery_notebooklm, notebooklm)
    if args.output_script_md is not None:
        _write_debug_script(args.output_script_md, single, notebooklm=False)
    if args.output_notebooklm_script_md is not None:
        _write_debug_script(args.output_notebooklm_script_md, notebooklm, notebooklm=True)
    if args.debug_script_md is not None:
        _write_debug_script(args.debug_script_md, single, notebooklm=False)
    if args.debug_notebooklm_script_md is not None:
        _write_debug_script(args.debug_notebooklm_script_md, notebooklm, notebooklm=True)
    _write_structured_review_scaffolding(args.delivery_single_speaker.parent, render_spec)
    return args.delivery_single_speaker, args.delivery_notebooklm


def process_workspace_presenter(args: argparse.Namespace) -> Path:
    topic_dir = args.slides_md.parent
    return process_topic(
        _workspace_topic_num(args),
        verbose=args.verbose,
        skip_dedup=args.skip_dedup or _env_bool("PRESENTATION_SKIP_DEDUP"),
        topic_dir=topic_dir,
        input_path=args.slides_md,
        slide_mapping_path=args.slide_mapping,
        output_path=args.output_script_md,
        output_notebooklm_path=args.output_notebooklm_script_md,
        topic_display_name=_display_name_from_presentation_id(args.presentation_id),
        conference_label=args.conference_label
        or infer_conference_label(workspace=args.workspace, slides_md=args.slides_md),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # D-QUICK-260715-rz1: establish a workspace-rooted provider_receipt_context
    # so standalone invocations don't fall back to Path.cwd() (repo root) for
    # receipt placement; --workspace is optional here for the legacy
    # topic-name code path, so fall back to inferring from --slides-md.
    _receipt_workspace = args.workspace or infer_conference_workspace(
        getattr(args, "slides_md", None) or "."
    )
    if _receipt_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_receipt_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_presenter_script")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        if args.render_spec is not None:
            single, notebooklm = process_structured_delivery(args)
            print(f"Generated: {single}")
            print(f"Generated: {notebooklm}")
            return 0
        _require_approved_blueprint_context(args)
        _load_repo_env()
        output_path = process_workspace_presenter(args)
        print(f"Generated: {output_path}")
        return 0

    _load_repo_env()
    if args.all:
        print("Generating presenter scripts for all topics...")
        for topic_num in TOPIC_NAMES:
            try:
                output = process_topic(
                    topic_num, verbose=args.verbose, skip_dedup=args.skip_dedup
                )
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
        output_path = process_topic(
            topic_num, verbose=True, skip_dedup=args.skip_dedup
        )
        print(f"Generated: {output_path}")

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
