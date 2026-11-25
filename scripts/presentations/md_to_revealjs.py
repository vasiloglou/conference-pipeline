#!/usr/bin/env python3
"""Convert Markdown slides to a self-contained Reveal.js HTML presentation.

Produces a single .html file that:
- Loads Reveal.js + plugins from CDN (no local install needed)
- Embeds SVG images inline (no broken links)
- Incorporates RelationalAI branding from the PPTX template
- Uses speaker notes (press 'S' to open speaker view)
- Supports PDF export via ?print-pdf query parameter
- Adds data-markdown-slide attributes for slide number mapping
- Exports a slide mapping JSON for video generation

Output files:
- presentation.html — the main HTML presentation
- presentation_slide_mapping.json — maps markdown slide numbers to HTML section indices
  (used by build_presenter_video.py to ensure slides sync with narration)

Usage:
    python scripts/presentations/md_to_revealjs.py \\
        --input path/to/5_draft_slides_with_images.md \\
        --output path/to/presentation.html

    # With explicit template path:
    python scripts/presentations/md_to_revealjs.py \\
        --input path/to/5_draft_slides_with_images.md \\
        --template /path/to/RelationalAI-template.pptx

PDF export:
    1. Open presentation.html?print-pdf in Chrome
    2. Ctrl/Cmd+P -> Save as PDF
    3. Set margins to None, enable Background graphics

Slide mapping:
    The slide mapping JSON is required for video generation because markdown slides
    don't map 1:1 to HTML sections. Content slides with figures generate TWO sections:
    - Main slide (content + small figure on the right)
    - Zoom companion slide (figure fills the screen)

    Example mapping: {"1": 0, "2": 2, "3": 3, "4": 5, ...}
    - Slide 1 → HTML index 0 (has zoom companion at index 1)
    - Slide 2 → HTML index 2 (no zoom companion)
    - Slide 3 → HTML index 3 (has zoom companion at index 4)
    - Slide 4 → HTML index 5 (has zoom companion at index 6)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path
from zipfile import ZipFile

# ---------------------------------------------------------------------------
# Template asset extraction
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.structured_video import render_spec_to_html

MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."

DEFAULT_TEMPLATE_PATHS = [
    PROJECT_ROOT / "conferences" / "neurips2025" / "decks" / "templates" / "RelationalAI.pptx",
    PROJECT_ROOT / "conferences" / "neurips2025" / "decks" / "templates" / "RelationalAI-template.pptx",
    PROJECT_ROOT / "conferences" / "neurips2025" / "presentations" / "templates" / "RelationalAI.pptx",
    PROJECT_ROOT / "conferences" / "neurips2025" / "presentations" / "templates" / "RelationalAI-template.pptx",
    Path.home() / "Downloads" / "RelationalAI.pptx",
    Path.home() / "Downloads" / "RelationalAI-template.pptx",
]

# Images inside the PPTX ZIP. These are the media names used by the
# NeurIPS 2025 RelationalAI presentation template; do not map arbitrary
# corporate theme decks to these roles just because they share a filename.
TEMPLATE_IMAGES = {
    "decorative_art": "ppt/media/image9.png",   # 1162x1183 decorative art (title slide)
    "logo":           "ppt/media/image8.png",    # 1101x158 RelationalAI logo (title slide)
    "bug":            "ppt/media/image12.png",   # 107x78 small brand symbol (content slides)
    "section_bg":     "ppt/media/image11.jpg",   # 1440x810 background photo (section dividers)
}

BRAND_ASSET_DIR = PROJECT_ROOT / "assets" / "brand" / "relationalai"
BRAND_ASSETS = {
    "logo": BRAND_ASSET_DIR / "logo_white.png",
    "decorative_art": BRAND_ASSET_DIR / "title_decorative_art.png",
    "bug": BRAND_ASSET_DIR / "bug_orange.png",
}
BRAND_REFERENCE_NAME = "relationalai-neurips2025-reference"
TEMPLATE_ENV_VARS = ("RAI_PRESENTATION_TEMPLATE", "RELATIONALAI_PRESENTATION_TEMPLATE")
TEMPLATE_REQUIRED_MESSAGE = (
    "RelationalAI NeurIPS 2025 brand assets are required for manifest-backed "
    "presentation render. Restore assets/brand/relationalai/*, set "
    "RAI_PRESENTATION_TEMPLATE to the NeurIPS RelationalAI template, or pass "
    "--template."
)


def relationalai_theme_contract() -> dict[str, object]:
    """Return the render-spec contract for the full RelationalAI theme path."""

    return {
        "renderer_path": "scripts/presentations/md_to_revealjs.py",
        "renderer_entrypoint": "md_to_revealjs",
        "theme_reference": "Full RelationalAI NeurIPS 2025 renderer path",
        "brand_reference_name": BRAND_REFERENCE_NAME,
        "template_images": dict(TEMPLATE_IMAGES),
        "brand_assets": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in BRAND_ASSETS.items()},
        "brand_assets_sha256": brand_assets_sha256(),
        "title_slide": {
            "required": True,
            "layout": "RelationalAI Title_Dark",
            "background": "midnight navy",
            "content_max_width_percent": 65,
        },
        "title_logo": {
            "required": True,
            "description": "RelationalAI title logo",
            "safe_area_css_px": {"width": 309, "height": 48, "minimum": "48x48 CSS px"},
            "position": {"top_css_px": 26, "left_css_px": 77},
            "background_size": "contain",
        },
        "decorative_art": {
            "required": True,
            "description": "RelationalAI title decorative art",
            "safe_area_css_px": {"width": 520, "height": 530},
            "position": {"top_css_px": -30, "right_css_px": -30},
            "opacity": 0.8,
        },
        "color_palette": {
            "required": ["navy", "rai-blue", "ice-blue", "rai-teal", "rai-orange", "rai-green"],
            "css_variables": {
                "--navy": "#1a1a2e",
                "--rai-blue": "#058DC7",
                "--ice-blue": "#a8d8ea",
                "--rai-teal": "#158158",
                "--rai-orange": "#ED561B",
                "--rai-green": "#50B432",
            },
        },
        "content_brand_bug": {
            "required": True,
            "description": "RelationalAI content brand bug",
            "safe_area_css_px": {"width": 54, "height": 40, "minimum": "32x32 CSS px"},
            "position": {"right_css_px": 24, "bottom_css_px": 18},
            "background_size": "contain",
        },
        "content_header": {
            "top_right_logo": False,
            "title_max_width_percent": 100,
            "description": "Content and bookend headings use the full inner slide width across conferences.",
        },
        "section_styling": {
            "required": True,
            "description": "RelationalAI section styling",
            "layout": "Impact1 section divider with optional section background image",
        },
        "readable_source_chips": {
            "required": True,
            "description": "readable source chips",
            "font_size_em": 0.36,
            "text_color": "var(--ice-blue)",
            "background": "rgba(168, 216, 234, 0.08)",
        },
        "uncropped_logos": {
            "required": True,
            "description": "uncropped logos use background-size contain and explicit safe areas",
        },
        "coverage_thresholds": {
            "alpha_non_background_min": 0.20,
            "luminance_standard_deviation_min": 4.0,
            "title_logo_region": "48x48 CSS px",
            "content_brand_bug_region": "32x32 CSS px",
        },
    }


def extract_template_assets(template_path: Path) -> dict[str, str]:
    """Extract key images from the RelationalAI PPTX template as base64 data URIs."""
    assets = {}
    if not template_path or not template_path.exists():
        return assets

    try:
        with ZipFile(template_path, 'r') as z:
            for key, zip_path in TEMPLATE_IMAGES.items():
                try:
                    data = z.read(zip_path)
                except KeyError:
                    continue
                assets[key] = _image_data_uri(zip_path, data)
            media_files = sorted(
                name
                for name in z.namelist()
                if name.startswith("ppt/media/") and name.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            if "section_bg" not in assets:
                for name in media_files:
                    if name.lower().endswith((".jpg", ".jpeg")):
                        assets["section_bg"] = _image_data_uri(name, z.read(name))
                        break
    except Exception as e:
        print(f"  Warning: could not read template {template_path}: {e}")

    return assets


def _image_data_uri(zip_path: str, data: bytes) -> str:
    ext = zip_path.rsplit('.', 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    b64 = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{b64}"


def load_brand_reference_assets() -> dict[str, str]:
    """Load bundled RelationalAI NeurIPS 2025 brand assets as data URIs."""
    assets: dict[str, str] = {}
    for key, path in BRAND_ASSETS.items():
        if path.exists():
            assets[key] = _image_data_uri(str(path), path.read_bytes())
    return assets


def brand_assets_sha256() -> str:
    h = hashlib.sha256()
    for key in sorted(BRAND_ASSETS):
        path = BRAND_ASSETS[key]
        if not path.exists():
            return ""
        h.update(key.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _template_candidates(workspace: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    for env_var in TEMPLATE_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())
    if workspace is not None:
        candidates.extend(
            [
                workspace / "decks" / "templates" / "RelationalAI.pptx",
                workspace / "decks" / "templates" / "RelationalAI-template.pptx",
                workspace / "templates" / "RelationalAI.pptx",
                workspace / "templates" / "RelationalAI-template.pptx",
            ]
        )
    candidates.extend(DEFAULT_TEMPLATE_PATHS)
    cloud_root = Path.home() / "Library" / "CloudStorage"
    if cloud_root.exists():
        candidates.extend(cloud_root.glob("GoogleDrive-*/My Drive/RelationalAI.pptx"))
        candidates.extend(cloud_root.glob("GoogleDrive-*/My Drive/RelationalAI-template.pptx"))
    return candidates


def find_template(workspace: Path | None = None) -> Path | None:
    """Auto-detect the RelationalAI template PPTX."""
    for p in _template_candidates(workspace):
        if p.exists():
            return p.resolve()
    return None


def template_sha256(template_path: Path | None) -> str:
    if template_path is None or not template_path.exists():
        return ""
    h = hashlib.sha256()
    with template_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

def parse_slides(md_text: str) -> list[dict]:
    """Parse the markdown file into slide dicts."""
    slides = []
    raw_blocks = re.split(r'\n---\n', md_text)

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        slide_match = re.search(r'^## Slide (\d+):\s*(.+)', block, re.MULTILINE)
        if not slide_match:
            continue

        slide_num = int(slide_match.group(1))
        slide_title = slide_match.group(2).strip()

        # Extract HTML comment metadata — find ALL comments, identify
        # the metadata block (contains SLIDE_TYPE, SPEAKER_NOTES, etc.).
        # Non-metadata comments (e.g. FIGURE_DESCRIPTION) are preserved
        # as content so process_content() can handle them.
        metadata = {}
        all_comments = list(re.finditer(r'<!--(.*?)-->', block, re.DOTALL))

        metadata_comment = None
        for cm in all_comments:
            body = cm.group(1)
            if re.search(r'^[A-Z_]+:', body, re.MULTILINE):
                _parse_metadata(body, metadata)
                metadata_comment = cm

        slide_type = metadata.get("SLIDE_TYPE", "content")
        if slide_type == "banner":
            slide_type = "section"
        elif slide_type == "quote":
            slide_type = "content"  # quote slides render as content
        section_name = metadata.get("SECTION", "")
        speaker_notes = metadata.get("SPEAKER_NOTES", "")

        # Content: everything after the title except the metadata comment.
        # Programmatic figure generation appends images after metadata, and
        # those images must remain visible in rendered/QC slides.
        content_start = slide_match.end()
        if metadata_comment:
            content = (
                block[content_start:metadata_comment.start()]
                + "\n"
                + block[metadata_comment.end():]
            ).strip()
        else:
            content = block[content_start:].strip()

        slides.append({
            "num": slide_num,
            "title": slide_title,
            "type": slide_type,
            "section": section_name,
            "content": content,
            "notes": speaker_notes,
            "metadata": metadata,
        })

    return slides


def _parse_metadata(comment_text: str, out: dict):
    lines = comment_text.strip().split('\n')
    current_key = None
    current_val = []
    for line in lines:
        line = line.strip()
        m = re.match(r'^([A-Z_]+):\s*(.*)', line)
        if m:
            if current_key:
                out[current_key] = ' '.join(current_val).strip()
            current_key = m.group(1)
            current_val = [m.group(2)]
        elif current_key:
            current_val.append(line)
    if current_key:
        out[current_key] = ' '.join(current_val).strip()


# ---------------------------------------------------------------------------
# Content processing
# ---------------------------------------------------------------------------

def process_content(content: str, base_dir: Path) -> str:
    """Process slide content: inline SVGs, convert markdown to HTML fragments."""
    content = _strip_internal_reference_keys(content)
    lines = content.split('\n')
    out_lines = []

    for line in lines:
        # Strip inline HTML comments (e.g. <!-- FIGURE_DESCRIPTION: ... -->)
        if re.match(r'^\s*<!--.*?-->\s*$', line, re.DOTALL):
            continue

        ref_match = re.match(r'^\s*-\s*References:\s*(.+?)\s*$', line)
        if ref_match:
            chips = _reference_chips_html(ref_match.group(1))
            if chips:
                out_lines.append(chips)
            continue

        # Inline SVG images
        img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', line.strip())
        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)
            svg_html = _inline_svg(img_path, alt_text, base_dir)
            if svg_html:
                out_lines.append(svg_html)
                continue
            # Fallback: keep as img tag
            out_lines.append(f'<img src="{img_path}" alt="{alt_text}" class="slide-figure">')
            continue

        out_lines.append(line)

    return '\n'.join(out_lines)


def _strip_internal_reference_keys(content: str) -> str:
    """Remove internal source keys that are not audience-facing citations."""

    cleaned = re.sub(r"\s*\((?:paper_abstract|source|chunk|paper_store|reference):[A-Za-z0-9_.:-]+\)", "", content)
    cleaned = re.sub(r"\s*\[(?:paper_abstract|source|chunk|paper_store|reference):[A-Za-z0-9_.:-]+\]", "", cleaned)
    return cleaned


def _reference_chips_html(reference_text: str) -> str:
    """Render long Markdown reference bullets as compact source chips."""

    matches = list(re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", reference_text))
    if not matches:
        label = _compact_reference_label(reference_text)
        title = html.escape(reference_text.strip() or label, quote=True)
        return f'<div class="reference-chips"><span>Sources</span><em title="{title}">{html.escape(label)}</em></div>'

    chips = []
    for match in matches[:3]:
        raw_label = match.group(1)
        url = match.group(2)
        label = _reference_link_label(raw_label)
        title = html.escape(raw_label.strip() or label, quote=True)
        chips.append(
            f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener" title="{title}" aria-label="{title}">{html.escape(label)}</a>'
        )
    return '<div class="reference-chips"><span>Sources</span>' + ''.join(chips) + '</div>'


def _reference_link_label(value: str) -> str:
    """Return the human-readable source title for visible link text."""

    label = re.sub(r"\s+", " ", value).strip()
    return _compact_reference_label(label or value)


def _compact_reference_label(value: str) -> str:
    known = _known_reference_label(value)
    if known:
        return known
    cleaned = re.sub(r"https?://\S+", "", value)
    cleaned = re.sub(r"[^A-Za-z0-9+/% -]+", " ", cleaned)
    words = [word for word in cleaned.split() if word]
    stop = {
        "a",
        "an",
        "and",
        "for",
        "of",
        "the",
        "to",
        "towards",
        "with",
    }
    kept = [word for word in words if word.lower() not in stop][:2]
    label = " ".join(kept or words[:3]).strip()
    return _ellipsize_reference_label(label) or "Source"


def _ellipsize_reference_label(label: str, *, limit: int = 18) -> str:
    compact = re.sub(r"\s+", " ", label).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 3)].rstrip() + "..."


def _known_reference_label(value: str) -> str:
    lower = value.lower()
    known = (
        ("cybench", "CyBench"),
        ("episodic", "Episodic Memory"),
        ("mmworld", "MMWorld"),
        ("star", "STAR"),
        ("second opinion", "Second Opinion"),
        ("selective rationale", "Second Opinion"),
        ("scbench", "SCBench"),
        ("helmet", "HELMET"),
        ("retrieval head", "Retrieval Heads"),
        ("livexiv", "LiveXiv"),
        ("aider", "Aider"),
        ("jax", "JAX"),
    )
    for needle, label in known:
        if needle in lower:
            return label
    return ""


def _inline_svg(img_path: str, alt_text: str, base_dir: Path) -> str | None:
    """Read an SVG file and return it as inline HTML."""
    p = Path(img_path)
    if not p.suffix.lower() == '.svg':
        return None

    abs_path = (base_dir / p).resolve()
    if not abs_path.exists():
        return None

    try:
        svg_content = abs_path.read_text('utf-8')
        # Strip XML declaration if present
        svg_content = re.sub(r'<\?xml[^?]*\?>\s*', '', svg_content)
        # Strip DOCTYPE declaration if present (invalid inside HTML body)
        svg_content = re.sub(r'<!DOCTYPE[^>]*>\s*', '', svg_content)
        # Wrap in a container div for sizing
        return f'<div class="svg-container" aria-label="{alt_text}">{svg_content}</div>'
    except Exception:
        return None


def _brand_img(
    assets: dict,
    key: str,
    *,
    class_name: str,
    role: str,
    alt: str = "",
    hidden: bool = False,
) -> str:
    src = str(assets.get(key) or "")
    if not src:
        return ""
    escaped_src = html.escape(src, quote=True)
    escaped_alt = html.escape(alt, quote=True)
    hidden_attrs = ' aria-hidden="true"' if hidden else ""
    return (
        f'<img class="{class_name}" data-qc-role="{role}" '
        f'src="{escaped_src}" alt="{escaped_alt}"{hidden_attrs}>'
    )


def _title_brand_html(assets: dict) -> str:
    return "\n".join(
        part
        for part in (
            _brand_img(
                assets,
                "logo",
                class_name="title-logo",
                role="title-logo",
                alt="RelationalAI logo",
            ),
            _brand_img(
                assets,
                "decorative_art",
                class_name="decorative-art",
                role="decorative-art",
                hidden=True,
            ),
        )
        if part
    )


def _content_brand_html(assets: dict) -> str:
    return _brand_img(
        assets,
        "bug",
        class_name="content-brand-bug",
        role="content-brand-bug",
        hidden=True,
    )


# ---------------------------------------------------------------------------
# Slide HTML generation
# ---------------------------------------------------------------------------

def slide_to_html(slide: dict, base_dir: Path, assets: dict | None = None) -> str:
    """Convert a slide dict to a Reveal.js <section> element."""
    assets = assets or {}
    stype = slide["type"]
    title = slide["title"]
    content = slide["content"]
    notes = slide["notes"]
    metadata = slide.get("metadata", {})

    # FIGURE_ZOOM metadata: "no" disables zoom, anything else (or absent) = auto
    figure_zoom = metadata.get("FIGURE_ZOOM", "").strip().lower()
    skip_zoom = figure_zoom == "no"

    # Process content (inline SVGs, etc.)
    processed = process_content(content, base_dir)

    # Build data attributes — all slides get the RelationalAI Midnight Executive background.
    attrs = ' data-background-color="#1a1a2e"'
    if stype == "section":
        section_bg = assets.get('section_bg', '')
        if section_bg:
            attrs += f' data-background-image="{section_bg}" data-background-size="cover" data-background-opacity="0.75"'

    # Notes block
    notes_html = ""
    if notes:
        escaped_notes = notes.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        notes_html = f'\n<aside class="notes">{escaped_notes}</aside>'

    # --- Type-specific rendering ---

    # Auto-promote content-heavy title slides to content layout
    if stype == "title":
        has_bullets = bool(re.search(r'^- ', content, re.MULTILINE))
        has_image = '![' in content
        line_count = len([l for l in content.split('\n') if l.strip()])
        if has_bullets or has_image or line_count > 6:
            stype = "content"  # fall through to content rendering

    if stype == "title":
        return _render_title(title, processed, attrs, notes_html, assets)
    elif stype == "section":
        return _render_section(title, processed, attrs, notes_html, assets, skip_zoom)
    else:
        return _render_content(title, processed, attrs, notes_html, stype, assets, skip_zoom)


def _title_text_for_layout(title: str, content: str) -> str:
    h1_match = re.search(r"^\s*#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    raw = h1_match.group(1) if h1_match else title
    raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
    raw = re.sub(r"[*_`#<>]", "", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _title_length_class(title: str, content: str) -> str:
    text = _title_text_for_layout(title, content)
    words = text.split()
    length = len(text)
    longest_word = max((len(word.strip(".,:;!?()[]{}")) for word in words), default=0)
    if length >= 100 or len(words) >= 14 or longest_word >= 20:
        return "title-xxlong"
    if length >= 72 or len(words) >= 10:
        return "title-xlong"
    if length >= 52 or len(words) >= 7:
        return "title-long"
    return "title-standard"


def _render_title(title: str, content: str, attrs: str, notes: str, assets: dict) -> str:
    title_class = _title_length_class(title, content)
    brand_html = _title_brand_html(assets)
    brand_prefix = f"{brand_html}\n" if brand_html else ""
    return f"""<section{attrs} class="slide-title {title_class}">
{brand_prefix}<div class="title-content" data-markdown>
<script type="text/template">
{content}
</script>
</div>{notes}
</section>"""


def _render_section(
    title: str,
    content: str,
    attrs: str,
    notes: str,
    assets: dict,
    skip_zoom: bool = False,
) -> str:
    # Clean up "Section Divider -- " prefix from title
    clean_title = re.sub(r'^Section Divider\s*[-:]+\s*', '', title, flags=re.IGNORECASE).strip()

    has_svg = '<div class="svg-container"' in content

    if has_svg:
        # --- Section with SVG: side-by-side layout + zoom companion ---
        md_parts = []
        svg_parts = []
        for line in content.split('\n'):
            if '<div class="svg-container"' in line or (svg_parts and not line.startswith('-') and not line.startswith('|') and not line.startswith('>') and not line.startswith('*') and not re.match(r'^\d+\.', line)):
                if '<div class="svg-container"' in line:
                    svg_parts.append(line)
                elif svg_parts and '</div>' not in svg_parts[-1]:
                    svg_parts.append(line)
                else:
                    md_parts.append(line)
            else:
                md_parts.append(line)

        md_content = '\n'.join(md_parts).strip()
        svg_content = '\n'.join(svg_parts).strip()

        # Build title: use H1 for section feel
        if md_content and re.match(r'^#\s', md_content):
            body = md_content
        else:
            body = f'# {clean_title}'
            if md_content:
                body += f'\n\n{md_content}'

        svg_with_id = svg_content.replace(
            '<div class="svg-container"',
            '<div class="svg-container" data-id="section-fig"',
            1
        )

        brand_html = _content_brand_html(assets)
        if skip_zoom:
            figure_class = "figure-float multi-figure" if svg_count >= 2 else "figure-float"
            return f"""<section{attrs} class="slide-section-with-figure" data-auto-animate>
{brand_html}
<div class="{figure_class}">
{svg_with_id}
</div>
<div class="section-inner" data-markdown>
<script type="text/template">
{body}
</script>
</div>{notes}
</section>"""

        # Main slide stays text-only when a zoom companion exists. Shrinking
        # dense SVGs into an inset made labels unreadable and caused false
        # text/figure overlap failures in visual QC.
        main_slide = f"""<section{attrs} class="slide-section" data-auto-animate>
{brand_html}
<div class="section-inner" data-markdown>
<script type="text/template">
{body}
</script>
</div>{notes}
</section>"""

        svg_count = svg_content.count('<div class="svg-container"')
        if svg_count >= 2:
            zoom_slide = f"""<section{attrs} class="slide-figure-zoom multi-figure-zoom" data-auto-animate data-auto-animate-duration="0.6">
{brand_html}
<div class="zoom-inner zoom-grid">
{svg_with_id}
</div>
</section>"""
        else:
            zoom_slide = f"""<section{attrs} class="slide-figure-zoom" data-auto-animate data-auto-animate-duration="0.6">
{brand_html}
<div class="zoom-inner">
{svg_with_id}
</div>
</section>"""

        return main_slide + '\n\n' + zoom_slide
    else:
        # --- Section without SVG: simple divider ---
        if content and re.match(r'^#\s', content):
            body = content
        else:
            body = f'# {clean_title}'
            if content:
                body += f'\n\n{content}'
        return f"""<section{attrs} class="slide-section">
{_content_brand_html(assets)}
<div class="section-inner" data-markdown>
<script type="text/template">
{body}
</script>
</div>{notes}
</section>"""


def _render_content(
    title: str,
    content: str,
    attrs: str,
    notes: str,
    stype: str,
    assets: dict,
    skip_zoom: bool = False,
) -> str:
    # Check if content has SVG containers (inline SVGs)
    has_svg = '<div class="svg-container"' in content
    layout_class = "slide-content"
    if has_svg:
        layout_class = "slide-content-with-figure"

    # Split content into markdown parts and SVG parts
    if has_svg:
        md_parts = []
        svg_parts = []
        for line in content.split('\n'):
            if '<div class="svg-container"' in line or (svg_parts and not line.startswith('-') and not line.startswith('|') and not line.startswith('>') and not line.startswith('*') and not re.match(r'^\d+\.', line)):
                if '<div class="svg-container"' in line:
                    svg_parts.append(line)
                elif svg_parts and '</div>' not in svg_parts[-1]:
                    svg_parts.append(line)
                else:
                    md_parts.append(line)
            else:
                md_parts.append(line)

        md_content = '\n'.join(md_parts).strip()
        svg_content = '\n'.join(svg_parts).strip()

        # Count how many SVG containers we have
        svg_count = svg_content.count('<div class="svg-container"')

        # Add data-id to each SVG container for auto-animate matching
        if svg_count == 1:
            svg_with_id = svg_content.replace(
                '<div class="svg-container"',
                '<div class="svg-container" data-id="fig"',
                1
            )
        else:
            # Multiple SVGs — give each a unique data-id
            svg_with_id = svg_content
            for i in range(svg_count):
                svg_with_id = svg_with_id.replace(
                    '<div class="svg-container"',
                    f'<div class="svg-container" data-id="fig-{i+1}"',
                    1
                )

        brand_html = _content_brand_html(assets)
        if skip_zoom:
            figure_class = "figure-float multi-figure" if svg_count >= 2 else "figure-float"
            return f"""<section{attrs} class="slide-content-with-figure" data-auto-animate>
{brand_html}
<div class="{figure_class}">
{svg_with_id}
</div>
<div data-markdown>
<script type="text/template">
### {title}

{md_content}
</script>
</div>{notes}
</section>"""

        # Main slide stays text-only when a zoom companion exists. This avoids
        # unreadable insets and keeps QC focused on a projection-safe figure
        # slide instead of a thumbnail embedded beside dense bullets.
        main_slide = f"""<section{attrs} class="slide-content" data-auto-animate>
{brand_html}
<div data-markdown>
<script type="text/template">
### {title}

{md_content}
</script>
</div>{notes}
</section>"""

        if svg_count == 1:
            # Single figure zoom
            zoom_slide = f"""<section{attrs} class="slide-figure-zoom" data-auto-animate data-auto-animate-duration="0.6">
{brand_html}
<div class="zoom-inner">
{svg_with_id}
</div>
</section>"""
        else:
            # Multi-figure zoom — side by side in zoom view
            zoom_slide = f"""<section{attrs} class="slide-figure-zoom multi-figure-zoom" data-auto-animate data-auto-animate-duration="0.6">
{brand_html}
<div class="zoom-inner zoom-grid">
{svg_with_id}
</div>
</section>"""

        return main_slide + '\n\n' + zoom_slide
    else:
        return f"""<section{attrs} class="{layout_class}">
{_content_brand_html(assets)}
<div data-markdown>
<script type="text/template">
### {title}

{content}
</script>
</div>{notes}
</section>"""


# ---------------------------------------------------------------------------
# Full HTML template
# ---------------------------------------------------------------------------

REVEAL_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="presentation-template" content="{template_name}">
<meta name="presentation-template-sha256" content="{template_sha256}">
<title>{title}</title>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/black.css" id="theme">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/highlight/monokai.css">

<style>
  /* --- RelationalAI + Midnight Executive Theme --- */
  :root {{
    --navy: #1a1a2e;
    --rai-blue: #058DC7;
    --ice-blue: #a8d8ea;
    --rai-teal: #158158;
    --rai-orange: #ED561B;
    --rai-green: #50B432;
    --light-gray: #e0e0e0;
    --mid-gray: #888;
  }}

  .reveal {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  }}

  /* Default slide background */
  .reveal .slides section {{
    background-color: var(--navy);
    color: var(--light-gray);
    text-align: left;
    padding: 30px 50px;
    box-sizing: border-box;
    overflow: hidden;          /* clip content that exceeds slide bounds */
  }}

  /* ================================================================
     TITLE SLIDE — Matches RelationalAI Title_Dark layout
     ================================================================ */
  .slide-title {{
    text-align: left !important;
    padding: 40px 60px !important;
  }}
  /* RelationalAI logo — top left (scoped to section to avoid Reveal.js background duplication) */
  .reveal .slides section.slide-title::before {{
    content: '';
    position: absolute;
    top: 26px;
    left: 77px;
    width: 309px;
    height: 48px;
    background-image: url('{logo_uri}');
    background-size: contain;
    background-repeat: no-repeat;
    z-index: 2;
  }}
  body.template-logo-missing .reveal .slides section.slide-title::before {{
    background-image: none;
  }}
  .reveal .slides section.slide-title .title-logo {{
    position: absolute;
    top: 26px;
    left: 77px;
    width: 309px;
    height: 48px;
    object-fit: contain;
    z-index: 3;
  }}
  /* Decorative art — top right */
  .reveal .slides section.slide-title::after {{
    content: '';
    position: absolute;
    top: -30px;
    right: -30px;
    width: 520px;
    height: 530px;
    background-image: url('{decorative_art_uri}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: top right;
    opacity: 0.8;
    z-index: 1;
    pointer-events: none;
  }}
  .reveal .slides section.slide-title .decorative-art {{
    position: absolute;
    top: -30px;
    right: -30px;
    width: 520px;
    height: 530px;
    opacity: 0.8;
    object-fit: contain;
    z-index: 2;
    pointer-events: none;
  }}
  .slide-title .title-content {{
    position: relative;
    z-index: 3;
    margin-top: 160px;
    max-width: 65%;
  }}
  .slide-title h1 {{
    font-size: 2.2em;
    color: white;
    margin-bottom: 0.1em;
    line-height: 1.1;
    overflow-wrap: break-word;
    text-wrap: balance;
  }}
  .slide-title.title-long .title-content {{
    margin-top: 132px;
    max-width: 63%;
  }}
  .slide-title.title-long h1 {{
    font-size: 1.92em;
    line-height: 1.05;
  }}
  .slide-title.title-xlong .title-content {{
    margin-top: 104px;
    max-width: 62%;
  }}
  .slide-title.title-xlong h1 {{
    font-size: 1.62em;
    line-height: 1.03;
  }}
  .slide-title.title-xxlong .title-content {{
    margin-top: 82px;
    max-width: 60%;
  }}
  .slide-title.title-xxlong h1 {{
    font-size: 1.38em;
    line-height: 1.02;
  }}
  .slide-title h2 {{
    font-size: 1.3em;
    color: var(--ice-blue);
    font-weight: 300;
    margin-top: 0.1em;
  }}
  .slide-title em {{
    color: var(--rai-orange);
    font-size: 0.8em;
    display: block;
    margin-top: 0.6em;
  }}
  .slide-title p {{
    color: var(--mid-gray);
    font-size: 0.7em;
  }}

  /* ================================================================
     SECTION DIVIDER — Matches RelationalAI Impact1 layout
     ================================================================ */
  .slide-section {{
    text-align: center !important;
    padding: 40px 60px !important;
    background-color: transparent !important;
  }}
  .slide-section .section-inner {{
    position: relative;
    z-index: 1;
    margin-top: 3%;
  }}
  /* Section with figure — side-by-side layout like content slides */
  .slide-section-with-figure {{
    text-align: left !important;
  }}
  .slide-section-with-figure .figure-float {{
    float: right;
    width: 42%;
    margin: 0 0 10px 16px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .slide-section-with-figure .section-inner {{
    margin-top: 0;
  }}
  .slide-section-with-figure .svg-container {{
    max-height: 520px;
  }}
  .slide-section-with-figure .svg-container svg {{
    max-height: 500px;
    max-width: 100%;
  }}
  .slide-section-with-figure h1 {{
    font-size: 1.3em;
    margin-bottom: 0.15em;
  }}
  .slide-section-with-figure li {{
    font-size: 0.48em;
    line-height: 1.25;
    margin-bottom: 0.04em;
    color: var(--light-gray);
  }}
  .slide-section-with-figure p {{
    font-size: 0.48em;
    line-height: 1.25;
  }}
  /* Plain section (no figure) — inline SVG fallback */
  .slide-section:not(.slide-section-with-figure) .svg-container {{
    max-height: 320px;
    margin-top: 8px;
  }}
  .slide-section:not(.slide-section-with-figure) .svg-container svg {{
    max-height: 310px;
  }}
  .slide-section h1 {{
    font-size: 1.68em;
    color: white;
    margin-bottom: 0.22em;
    text-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }}
  .slide-section ol, .slide-section ul {{
    display: inline-block;
    text-align: left;
    font-size: 0.64em;
    color: var(--ice-blue);
  }}
  .slide-section li {{
    margin-bottom: 0.08em;
    text-shadow: 0 1px 4px rgba(0,0,0,0.4);
  }}
  .slide-section em {{
    color: var(--rai-orange);
    font-size: 0.8em;
    display: block;
    margin-top: 0.5em;
    text-shadow: 0 1px 4px rgba(0,0,0,0.4);
  }}
  .slide-section strong {{
    color: white;
  }}

  /* ================================================================
     CONTENT SLIDE — Full-width text
     ================================================================ */
  .slide-content {{
  }}
  .slide-content h3 {{
    font-size: 1.15em;
    color: var(--ice-blue);
    margin-bottom: 0.4em;
    border-bottom: 2px solid var(--rai-orange);
    padding-bottom: 0.15em;
  }}
  .slide-content p {{
    font-size: 0.60em;
    line-height: 1.35;
    margin: 0.15em 0;
  }}
  .slide-content li {{
    font-size: 0.58em;
    line-height: 1.25;
    margin-bottom: 0.06em;
    color: var(--light-gray);
  }}
  .slide-content strong {{
    color: white;
  }}
  .content-brand-bug {{
    position: absolute;
    right: 24px;
    bottom: 18px;
    width: 54px;
    height: 40px;
    object-fit: contain;
    opacity: 0.82;
    pointer-events: none;
  }}
  .slide-content a {{
    color: var(--ice-blue);
    text-decoration: underline;
  }}
  .reference-chips {{
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 5px;
    margin-top: 0.22em;
    max-width: 100%;
    font-size: 0.36em;
    line-height: 1.18;
  }}
  .reference-chips span {{
    color: #d9f2fb;
    text-transform: uppercase;
    letter-spacing: 0;
    font-size: 0.84em;
    font-weight: 700;
    margin-right: 2px;
    padding: 4px 7px 0 0;
  }}
  .reference-chips a,
  .reference-chips em {{
    display: inline-block;
    padding: 3px 8px;
    border: 1px solid rgba(168, 216, 234, 0.35);
    border-radius: 4px;
    color: var(--ice-blue);
    background: rgba(168, 216, 234, 0.08);
    text-decoration: none;
    font-style: normal;
    white-space: nowrap;
  }}
  .slide-content blockquote {{
    border-left: 4px solid var(--rai-orange);
    padding: 0.2em 0.6em;
    margin: 0.3em 0;
    font-style: italic;
    font-size: 0.58em;
    color: var(--ice-blue);
    background: rgba(168, 216, 234, 0.05);
  }}
  /* RelationalAI brand bug — small, fixed, and outside the reading column. */
  .reveal .slides section.slide-content::after {{
    content: '';
    position: absolute;
    right: 24px;
    bottom: 18px;
    width: 54px;
    height: 40px;
    background-image: url('{bug_uri}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.82;
    pointer-events: none;
  }}
  body.template-bug-missing .reveal .slides section.slide-content::after {{
    background-image: none;
  }}

  /* ================================================================
     CONTENT + FIGURE LAYOUT — Two-column
     ================================================================ */
  .reveal .slides section.slide-content-with-figure {{
    min-height: 580px !important;  /* prevent too-short slides */
  }}
  .slide-content-with-figure h3 {{
    font-size: 0.95em;
    color: var(--ice-blue);
    margin-bottom: 0.25em;
    border-bottom: 2px solid var(--rai-orange);
    padding-bottom: 0.1em;
    overflow: hidden;          /* creates BFC so title + border stop before the float */
    max-width: 55%;            /* prevent title from extending into figure area */
  }}
  /* Figure floats right so bullets wrap around it */
  .slide-content-with-figure .figure-float {{
    float: right;
    width: 42%;
    margin: 0 0 10px 16px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  /* Override Reveal.js inline-block on ul/ol so text wraps around float */
  .slide-content-with-figure ul,
  .slide-content-with-figure ol {{
    display: block !important;
    width: auto !important;
    text-align: left;
  }}
  .slide-content-with-figure li {{
    /* Reduced from 0.49em/1.24 -- full-sentence bullets (as opposed to short
       phrase bullets) were overflowing the fixed 1280x720 slide canvas and
       getting silently clipped by section {{ overflow: hidden }}, caught by
       visual QC as truncated/illegible text. This template has no
       content-aware autofit, so the fix is a smaller static ceiling that
       gives headroom for 2-3 full sentences instead of short phrases. */
    font-size: 0.40em;
    line-height: 1.16;
    margin-bottom: 0.04em;
    color: var(--light-gray);
  }}
  .slide-content-with-figure strong {{
    color: white;
  }}
  .slide-content-with-figure a {{
    color: var(--ice-blue);
    text-decoration: underline;
  }}
  .slide-content-with-figure blockquote {{
    display: block !important;
    width: auto !important;
    border-left: 3px solid var(--rai-orange);
    padding: 0.15em 0.5em;
    font-style: italic;
    font-size: 0.48em;
    color: var(--ice-blue);
  }}
  .slide-content-with-figure table {{
    font-size: 0.42em;
  }}
  .slide-content-with-figure table th,
  .slide-content-with-figure table td {{
    padding: 0.15em 0.4em;
  }}
  .slide-content-with-figure p {{
    /* See the matching li{{}} comment above: reduced from 0.50em/1.3. */
    font-size: 0.42em;
    line-height: 1.2;
    margin: 0.1em 0;
  }}
  /* RelationalAI brand bug — small, fixed, and outside the reading column. */
  .reveal .slides section.slide-content-with-figure::after {{
    content: '';
    position: absolute;
    right: 24px;
    bottom: 18px;
    width: 54px;
    height: 40px;
    background-image: url('{bug_uri}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.82;
    pointer-events: none;
  }}
  body.template-bug-missing .reveal .slides section.slide-content-with-figure::after {{
    background-image: none;
  }}

  /* ================================================================
     MULTI-FIGURE: Vertical stack in content+figure right column
     ================================================================ */
  .slide-content-with-figure .figure-float.multi-figure {{
    flex-direction: column;
    gap: 6px;
  }}
  .slide-content-with-figure .figure-float.multi-figure .svg-container {{
    max-height: 240px;
  }}
  .slide-content-with-figure .figure-float.multi-figure .svg-container svg {{
    max-height: 230px;
  }}

  /* Multi-figure zoom — side by side */
  .multi-figure-zoom .zoom-inner.zoom-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: flex-start;
    gap: 16px;
    margin-top: 2%;
  }}
  .multi-figure-zoom .zoom-grid .svg-container {{
    width: 48% !important;
    max-width: 48%;
    max-height: 540px;
    flex: 0 1 48%;
  }}
  .multi-figure-zoom .zoom-grid .svg-container svg {{
    max-height: 520px;
    max-width: 100%;
  }}
  .multi-figure-zoom .zoom-grid .figure-caption {{
    flex: 0 0 100%;
  }}

  /* ================================================================
     SVG FIGURES
     ================================================================ */
  .svg-container {{
    width: 100%;
    max-height: 520px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .svg-container svg {{
    max-width: 100%;
    max-height: 500px;
    height: auto;
    width: auto;
  }}

  /* Stand-alone img fallback */
  img.slide-figure {{
    max-width: 100%;
    max-height: 500px;
  }}

  /* ================================================================
     FIGURE ZOOM SLIDE — full-screen figure via data-auto-animate
     ================================================================ */
  .slide-figure-zoom {{
    padding: 20px 40px !important;
    text-align: center !important;
  }}
  .slide-figure-zoom .zoom-inner {{
    margin-top: 2%;
  }}
  .slide-figure-zoom .svg-container {{
    max-height: 600px;
    max-width: 90%;
    margin: 0 auto;
  }}
  .slide-figure-zoom .svg-container svg {{
    max-height: 580px;
    max-width: 100%;
  }}
  .slide-figure-zoom .figure-caption {{
    text-align: center;
    font-size: 0.55em;
    color: var(--mid-gray);
    margin-top: 0.4em;
  }}
  .reveal .slides section.slide-figure-zoom::after {{
    content: '';
    position: absolute;
    right: 24px;
    bottom: 18px;
    width: 54px;
    height: 40px;
    background-image: url('{bug_uri}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.82;
    pointer-events: none;
  }}
  body.template-bug-missing .reveal .slides section.slide-figure-zoom::after {{
    background-image: none;
  }}

  /* ================================================================
     TABLES
     ================================================================ */
  .reveal table {{
    font-size: 0.55em;
    border-collapse: collapse;
    width: 100%;
    margin: 0.3em 0;
  }}
  .reveal table th {{
    background: var(--navy);
    color: var(--ice-blue);
    border-bottom: 2px solid var(--rai-orange);
    padding: 0.3em 0.5em;
    text-align: left;
    font-weight: 600;
  }}
  .reveal table td {{
    padding: 0.25em 0.5em;
    border-bottom: 1px solid rgba(168, 216, 234, 0.15);
    color: var(--light-gray);
  }}
  .reveal table tr:hover td {{
    background: rgba(168, 216, 234, 0.05);
  }}
  .reveal table a {{
    color: var(--ice-blue);
  }}

  /* ================================================================
     SLIDE NUMBERS
     ================================================================ */
  .reveal .slide-number {{
    color: var(--mid-gray);
    font-size: 0.5em;
  }}

  /* ================================================================
     PRINT / PDF
     ================================================================ */
  @media print {{
    .reveal .slides section {{
      width: 1280px !important;
      height: 720px !important;
      page-break-after: always;
      box-sizing: border-box;
    }}
    .reveal .controls {{
      display: none !important;
    }}
  }}
</style>
</head>
<body class="{template_status_classes}">

<div class="reveal">
<div class="slides">
{slides_html}
</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/markdown/markdown.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/highlight/highlight.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/notes/notes.js"></script>

<script>
  var hideControls = window.location.search.indexOf('hidecontrols') !== -1;
  Reveal.initialize({{
    hash: true,
    slideNumber: !hideControls,
    controls: !hideControls,
    width: 1280,
    height: 720,
    margin: 0.04,
    minScale: 0.2,
    maxScale: 2.0,
    transition: 'slide',
    transitionSpeed: 'default',
    backgroundTransition: 'fade',
    pdfMaxPagesPerSlide: 1,
    pdfSeparateFragments: false,
    plugins: [ RevealMarkdown, RevealHighlight, RevealNotes ]
  }});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_presentation(input_path: Path, output_path: Path,
                       template_path: Path | None = None,
                       verbose: bool = False,
                       slide_mapping_path: Path | None = None,
                       workspace: Path | None = None,
                       require_template: bool = False) -> Path:
    """Build a Reveal.js HTML presentation from markdown slides."""
    md_text = input_path.read_text('utf-8')
    base_dir = input_path.parent

    # Extract presentation title from first H1
    title_match = re.search(r'^# (.+?)$', md_text, re.MULTILINE)
    pres_title = title_match.group(1).strip() if title_match else "Presentation"

    # Extract template branding assets, then fill any missing required brand
    # roles from the checked-in NeurIPS 2025 reference render. This prevents a
    # same-named but visually incompatible PPTX from silently changing the deck.
    tpl = template_path or find_template(workspace)
    assets = {}
    template_assets = {}
    if tpl:
        template_assets = extract_template_assets(tpl)
        if template_assets and not all(key in template_assets for key in ("logo", "decorative_art", "bug")):
            if verbose:
                missing = sorted({"logo", "decorative_art", "bug"} - set(template_assets))
                print(f"  Template ignored for branding; missing required assets: {', '.join(missing)}")
            template_assets = {}
        assets = dict(template_assets)
        if verbose:
            print(f"  Template: {tpl.name} ({len(assets)} assets extracted)")
    elif verbose:
        print("  No template found — generating without branding")
    reference_assets = load_brand_reference_assets()
    for key in ("logo", "decorative_art", "bug"):
        if key not in assets and key in reference_assets:
            assets[key] = reference_assets[key]
    if require_template and not all(assets.get(key) for key in ("logo", "decorative_art", "bug")):
        raise FileNotFoundError(TEMPLATE_REQUIRED_MESSAGE)

    slides = parse_slides(md_text)

    if verbose:
        print(f"  Parsed {len(slides)} slides")
        types = {}
        for s in slides:
            types[s['type']] = types.get(s['type'], 0) + 1
        print(f"  Types: {types}")

    # Generate HTML for each slide
    sections = []
    slide_mapping = {}  # {markdown_slide_num: html_section_index}
    html_section_index = 0

    for i, slide in enumerate(slides):
        if verbose:
            img_count = len(re.findall(r'!\[', slide['content']))
            has_table = '|' in slide['content'] and '---|' in slide['content']
            indicators = ""
            if img_count:
                indicators += f" [{img_count} img]"
            if has_table:
                indicators += " [table]"
            print(f"  [{i+1}/{len(slides)}] Slide {slide['num']}: "
                  f"{slide['type']}{indicators} - {slide['title'][:55]}")

        html = slide_to_html(slide, base_dir, assets)

        # Track this markdown slide number -> first HTML section index
        slide_num = slide['num']
        slide_mapping[slide_num] = html_section_index

        # Count how many HTML sections this slide generated
        section_count = html.count('<section')

        # Add data-markdown-slide attribute to all sections from this slide
        modified_html = re.sub(
            r'<section([^>]*?)>',
            lambda m: f'<section data-markdown-slide="{slide_num}"{m.group(1)}>',
            html
        )

        sections.append(modified_html)
        html_section_index += section_count

    slides_html = '\n\n'.join(sections)
    template_status_classes = []
    if not assets.get("logo"):
        template_status_classes.append("template-logo-missing")
    if not assets.get("bug"):
        template_status_classes.append("template-bug-missing")
    template_name = tpl.name if tpl and template_assets else BRAND_REFERENCE_NAME
    template_hash = template_sha256(tpl) if tpl and template_assets else brand_assets_sha256()

    full_html = REVEAL_HTML_TEMPLATE.format(
        title=pres_title,
        slides_html=slides_html,
        logo_uri=assets.get('logo', ''),
        decorative_art_uri=assets.get('decorative_art', ''),
        bug_uri=assets.get('bug', ''),
        template_name=template_name,
        template_sha256=template_hash,
        template_status_classes=" ".join(template_status_classes),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_html, encoding='utf-8')

    # Write slide mapping JSON
    mapping_path = slide_mapping_path or output_path.parent / f"{output_path.stem}_slide_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(slide_mapping, f, indent=2, sort_keys=True)

    if verbose:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n  Saved: {output_path} ({size_mb:.1f} MB)")
        print(f"  Slide mapping: {mapping_path} ({len(slide_mapping)} slides)")
        print(f"  Open in browser to view. Press 'S' for speaker notes.")
        print(f"  For PDF: open {output_path.name}?print-pdf in Chrome, then Ctrl+P -> Save as PDF")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Markdown slides to Reveal.js HTML")
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--topic", help="Selected topic metadata for manifest-backed runs.")
    parser.add_argument("--render-spec", type=Path, help="Structured render_spec.json input.")
    parser.add_argument("--slides-md", "--input", "-i", dest="slides_md", type=Path, help="Markdown slides file.")
    parser.add_argument("--output-html", "--output", "-o", dest="output_html", type=Path, default=None, help="Output HTML file.")
    parser.add_argument("--slide-mapping", type=Path, help="Output slide mapping JSON file.")
    parser.add_argument("--template", "-t", type=str, default=None,
                        help="Path to NeurIPS RelationalAI PPTX template (optional if bundled brand assets exist)")
    parser.add_argument(
        "--require-template",
        dest="require_brand_assets",
        action="store_true",
        help="Deprecated alias for --require-brand-assets.",
    )
    parser.add_argument(
        "--require-brand-assets",
        dest="require_brand_assets",
        action="store_true",
        help="Fail closed unless exact RelationalAI NeurIPS reference branding is available.",
    )
    parser.add_argument(
        "--allow-missing-template",
        action="store_true",
        help="Permit unbranded render output. For tests/offline previews only.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned structured render export without writing files.")

    args = parser.parse_args(argv)

    if args.slides_md is None and args.render_spec is None:
        parser.error("--slides-md/--input or --render-spec is required")

    if args.workspace is not None:
        if args.render_spec is not None:
            required_fields = (
                ("--presentation-id", args.presentation_id),
                ("--presentation-type", args.presentation_type),
                ("--render-spec", args.render_spec),
                ("--output-html", args.output_html),
                ("--slide-mapping", args.slide_mapping),
            )
        else:
            required_fields = (
                ("--presentation-id", args.presentation_id),
                ("--presentation-type", args.presentation_type),
                ("--slides-md", args.slides_md),
                ("--output-html", args.output_html),
                ("--slide-mapping", args.slide_mapping),
            )
        missing = [flag for flag, value in required_fields if value is None or value == ""]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("render_spec", "--render-spec"),
            ("slides_md", "--slides-md"),
            ("output_html", "--output-html"),
            ("slide_mapping", "--slide-mapping"),
        ):
            if getattr(args, attr) is None:
                continue
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _render_spec_reference_chips(source_chips: list[object]) -> str:
    chips: list[str] = []
    for ref in source_chips[:3]:
        if not isinstance(ref, dict):
            continue
        label = html.escape(str(ref.get("title") or ref.get("label") or "Source").strip())
        url = html.escape(str(ref.get("url") or "#"), quote=True)
        chips.append(f'<a href="{url}" target="_blank" rel="noopener">{label}</a>')
    if not chips:
        chips.append("<em>Source-grounded evidence</em>")
    return '<div class="reference-chips"><span>Sources</span>' + "".join(chips) + "</div>"


def _render_spec_to_html(args: argparse.Namespace) -> Path:
    if args.slide_mapping is None:
        raise ValueError("--slide-mapping is required for render-spec HTML output")
    return render_spec_to_html(
        args.render_spec,
        output_html=Path(args.output_html),
        slide_mapping=Path(args.slide_mapping),
        workspace=Path(args.workspace) if args.workspace is not None else None,
        require_brand_assets=bool(args.require_brand_assets),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.render_spec is not None:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "output_html": str(args.output_html),
                        "render_spec": str(args.render_spec),
                        "slide_mapping": str(args.slide_mapping),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = _render_spec_to_html(args)
        print(f"Generated: {result}")
        return 0

    input_path = Path(args.slides_md)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    output_path = Path(args.output_html) if args.output_html else input_path.parent / "presentation.html"
    template_path = Path(args.template) if args.template else None
    slide_mapping_path = Path(args.slide_mapping) if args.slide_mapping else None
    require_template = args.require_brand_assets or (args.workspace is not None and not args.allow_missing_template)

    print(f"Converting: {input_path.name}")
    try:
        result = build_presentation(
            input_path,
            output_path,
            template_path,
            verbose=args.verbose,
            slide_mapping_path=slide_mapping_path,
            workspace=args.workspace,
            require_template=require_template,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Generated: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
