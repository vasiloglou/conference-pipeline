#!/usr/bin/env python3
"""Convert Markdown slides to PowerPoint presentation (v2).

Improvements over v1:
- Uses 6 template layouts (title, section, content, content+image, agenda, closing)
- Preserves bold, italic, and hyperlink formatting
- Handles markdown tables (renders as PPTX table shapes)
- Converts SVG images to PNG for embedding (via cairosvg)
- No artificial cap on bullet count
- Sub-bullet (level 1) support
- Speaker notes from HTML comment metadata
- Blockquote support

Usage:
    python scripts/presentations/md_to_pptx_v2.py \
        --input path/to/5_draft_slides_with_images.md \
        --template path/to/template.pptx \
        --output path/to/output.pptx -v
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default template locations (checked in order)
DEFAULT_TEMPLATES = [
    Path.home() / "Downloads" / "RelationalAI-template.pptx",
    PROJECT_ROOT / "conferences" / "neurips2025" / "presentations" / "templates" / "RelationalAI-template.pptx",
]

# Layout indices from the RelationalAI 53-layout template
LAYOUT_MAP = {
    "title": 50,         # Title_Dark
    "section": 3,        # Section -DC THEME
    "content": 13,       # Title | Content
    "content_image": 15, # Left Content image right Blue stripe
    "agenda": 2,         # Agenda -DC THEME
    "closing": 51,       # Impact1
}

# Fallback layout if a preferred index doesn't exist
FALLBACK_LAYOUT = 49  # Text_White_Bug

# Midnight Executive color palette
NAVY = RGBColor(0x1A, 0x1A, 0x2E)
ICE_BLUE = RGBColor(0xA8, 0xD8, 0xEA)
CORAL = RGBColor(0xFF, 0x6B, 0x6B)
GREEN = RGBColor(0x51, 0xCF, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RichText:
    text: str
    bold: bool = False
    italic: bool = False
    url: str | None = None


@dataclass
class BulletItem:
    segments: list[RichText]
    level: int = 0


@dataclass
class TableData:
    headers: list[list[RichText]]
    rows: list[list[list[RichText]]]
    col_count: int


@dataclass
class SlideData:
    slide_num: int
    title: str
    slide_type: str  # title, section, content, agenda, closing
    section_name: str = ""
    bullets: list[BulletItem] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    blockquotes: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    speaker_notes: str = ""
    raw_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Markdown Parser
# ---------------------------------------------------------------------------

# Inline formatting regex: bold-link, bold, italic, link
INLINE_RE = re.compile(
    r'\*\*\[([^\]]+)\]\(([^)]+)\)\*\*'  # **[text](url)**
    r'|\*\*([^*]+?)\*\*'                  # **text**
    r'|\*([^*]+?)\*'                       # *text*
    r'|\[([^\]]+)\]\(([^)]+)\)'           # [text](url)
)


def parse_rich_text(text: str) -> list[RichText]:
    """Parse inline markdown formatting into RichText segments."""
    segments: list[RichText] = []
    last_end = 0

    for m in INLINE_RE.finditer(text):
        # Plain text before match
        if m.start() > last_end:
            plain = text[last_end:m.start()]
            if plain:
                segments.append(RichText(text=plain))

        if m.group(1) is not None:      # **[text](url)**
            segments.append(RichText(text=m.group(1), bold=True, url=m.group(2)))
        elif m.group(3) is not None:     # **text**
            segments.append(RichText(text=m.group(3), bold=True))
        elif m.group(4) is not None:     # *text*
            segments.append(RichText(text=m.group(4), italic=True))
        elif m.group(5) is not None:     # [text](url)
            segments.append(RichText(text=m.group(5), url=m.group(6)))

        last_end = m.end()

    # Remaining text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            segments.append(RichText(text=remaining))

    if not segments:
        segments.append(RichText(text=text))

    return segments


def parse_markdown_slides(markdown_text: str) -> list[SlideData]:
    """Parse the full markdown file into a list of SlideData objects."""
    slides: list[SlideData] = []

    # Split on --- separators
    raw_blocks = re.split(r'\n---\n', markdown_text)

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Match slide header
        slide_match = re.search(r'^## Slide (\d+):\s*(.+)', block, re.MULTILINE)
        if not slide_match:
            continue

        slide_num = int(slide_match.group(1))
        slide_title = slide_match.group(2).strip()

        # Extract HTML comment metadata
        metadata = {}
        comment_match = re.search(r'<!--(.*?)-->', block, re.DOTALL)
        if comment_match:
            _parse_metadata(comment_match.group(1), metadata)

        slide_type = metadata.get("SLIDE_TYPE", "content")
        # Normalize: banner -> section (backward compat)
        if slide_type == "banner":
            slide_type = "section"

        section_name = metadata.get("SECTION", "")
        speaker_notes = metadata.get("SPEAKER_NOTES", "")

        # Get content lines (between title and comment block)
        content_start = slide_match.end()
        comment_start = comment_match.start() if comment_match else len(block)
        content_text = block[content_start:comment_start].strip()
        content_lines = content_text.split('\n') if content_text else []

        # Parse content elements
        bullets: list[BulletItem] = []
        tables: list[TableData] = []
        blockquotes: list[str] = []
        images: list[str] = []
        raw_lines: list[str] = []

        i = 0
        while i < len(content_lines):
            line = content_lines[i]
            stripped = line.strip()

            # Image reference
            img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', stripped)
            if img_match:
                images.append(img_match.group(2))
                i += 1
                continue

            # Table detection (line starts with |)
            if stripped.startswith('|') and i + 1 < len(content_lines):
                table_lines = [stripped]
                i += 1
                while i < len(content_lines) and content_lines[i].strip().startswith('|'):
                    table_lines.append(content_lines[i].strip())
                    i += 1
                table = _parse_table(table_lines)
                if table:
                    tables.append(table)
                continue

            # Blockquote
            if stripped.startswith('>'):
                quote_text = stripped.lstrip('> ').strip()
                # Clean markdown from quote
                quote_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', quote_text)
                quote_text = re.sub(r'\*([^*]+)\*', r'\1', quote_text)
                blockquotes.append(quote_text)
                i += 1
                continue

            # Bullet point
            bullet_match = re.match(r'^(\s*)[-*]\s+(.+)', stripped)
            if bullet_match:
                indent = len(line) - len(line.lstrip())
                level = 1 if indent >= 2 else 0
                text = bullet_match.group(2).strip()
                bullets.append(BulletItem(
                    segments=parse_rich_text(text),
                    level=level,
                ))
                i += 1
                continue

            # Numbered list
            num_match = re.match(r'^(\d+)\.\s+(.+)', stripped)
            if num_match:
                text = num_match.group(2).strip()
                bullets.append(BulletItem(
                    segments=parse_rich_text(text),
                    level=0,
                ))
                i += 1
                continue

            # Non-empty, non-heading line -> raw content
            if stripped and not stripped.startswith('#'):
                raw_lines.append(stripped)

            i += 1

        slides.append(SlideData(
            slide_num=slide_num,
            title=slide_title,
            slide_type=slide_type,
            section_name=section_name,
            bullets=bullets,
            tables=tables,
            blockquotes=blockquotes,
            images=images,
            speaker_notes=speaker_notes,
            raw_lines=raw_lines,
        ))

    return slides


def _parse_metadata(comment_text: str, out: dict):
    """Parse HTML comment metadata into a dict."""
    lines = comment_text.strip().split('\n')
    current_key = None
    current_val = []

    for line in lines:
        line = line.strip()
        m = re.match(r'^([A-Z_]+):\s*(.*)', line)
        if m:
            # Save previous key
            if current_key:
                out[current_key] = ' '.join(current_val).strip()
            current_key = m.group(1)
            current_val = [m.group(2)]
        elif current_key:
            current_val.append(line)

    if current_key:
        out[current_key] = ' '.join(current_val).strip()


def _parse_table(lines: list[str]) -> TableData | None:
    """Parse markdown table lines into a TableData object."""
    if len(lines) < 3:  # header + separator + at least one data row
        return None

    def split_cells(line: str) -> list[str]:
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        return [c.strip() for c in line.split('|')]

    # Check for separator row (---|---|---)
    sep_idx = None
    for idx, line in enumerate(lines):
        if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
            sep_idx = idx
            break

    if sep_idx is None or sep_idx == 0:
        return None

    header_cells = split_cells(lines[0])
    col_count = len(header_cells)
    headers = [parse_rich_text(c) for c in header_cells]

    rows = []
    for line in lines[sep_idx + 1:]:
        cells = split_cells(line)
        # Pad/truncate to col_count
        while len(cells) < col_count:
            cells.append("")
        cells = cells[:col_count]
        rows.append([parse_rich_text(c) for c in cells])

    return TableData(headers=headers, rows=rows, col_count=col_count)


# ---------------------------------------------------------------------------
# SVG Converter
# ---------------------------------------------------------------------------

class SVGConverter:
    """Convert SVG files to PNG for embedding in PPTX."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str | None:
        try:
            import cairosvg  # noqa: F401
            return "cairosvg"
        except ImportError:
            pass
        try:
            from svglib.svglib import svg2rlg  # noqa: F401
            from reportlab.graphics import renderPM  # noqa: F401
            return "svglib"
        except ImportError:
            pass
        return None

    def convert(self, svg_path: Path) -> Path | None:
        """Convert SVG to PNG. Returns PNG path or None on failure."""
        if self._backend is None:
            return None
        if not svg_path.exists():
            return None

        png_name = svg_path.stem + ".png"
        png_path = self.cache_dir / png_name

        # Cache check: skip if PNG is newer
        if png_path.exists() and png_path.stat().st_mtime > svg_path.stat().st_mtime:
            return png_path

        try:
            if self._backend == "cairosvg":
                import cairosvg
                cairosvg.svg2png(
                    url=str(svg_path),
                    write_to=str(png_path),
                    output_width=2000,
                )
            elif self._backend == "svglib":
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPM
                drawing = svg2rlg(str(svg_path))
                if drawing:
                    renderPM.drawToFile(drawing, str(png_path), fmt='PNG', dpi=200)
                else:
                    return None

            return png_path if png_path.exists() else None
        except Exception as e:
            print(f"  Warning: SVG conversion failed for {svg_path.name}: {e}", file=sys.stderr)
            return None


# ---------------------------------------------------------------------------
# Layout Mapper
# ---------------------------------------------------------------------------

class LayoutMapper:
    """Map slide types to template layouts with placeholder probing."""

    def __init__(self, prs: Presentation):
        self.prs = prs
        self._layout_count = len(prs.slide_layouts)
        # Verify which layouts exist
        self._available = {}
        for key, idx in LAYOUT_MAP.items():
            if idx < self._layout_count:
                self._available[key] = idx
            else:
                self._available[key] = min(FALLBACK_LAYOUT, self._layout_count - 1)

    def get_layout(self, slide_data: SlideData):
        """Return (layout_key, SlideLayout) for a slide."""
        stype = slide_data.slide_type

        if stype == "title":
            key = "title"
        elif stype == "section":
            key = "section"
        elif stype == "agenda":
            key = "agenda"
        elif stype == "closing":
            key = "closing"
        elif slide_data.images and "content_image" in self._available:
            key = "content_image"
        else:
            key = "content"

        idx = self._available.get(key, self._available["content"])
        return key, self.prs.slide_layouts[idx]


# ---------------------------------------------------------------------------
# Slide Renderer
# ---------------------------------------------------------------------------

class SlideRenderer:
    """Create PPTX slides from SlideData objects."""

    def __init__(self, prs: Presentation, layout_mapper: LayoutMapper,
                 svg_converter: SVGConverter | None, base_dir: Path, verbose: bool = False):
        self.prs = prs
        self.layout_mapper = layout_mapper
        self.svg_converter = svg_converter
        self.base_dir = base_dir
        self.verbose = verbose

    def render(self, slide_data: SlideData):
        """Render a single slide."""
        dispatch = {
            "title": self._render_title,
            "section": self._render_section,
            "agenda": self._render_content,
            "closing": self._render_closing,
        }
        method = dispatch.get(slide_data.slide_type, self._render_content)
        slide = method(slide_data)
        self._add_speaker_notes(slide, slide_data.speaker_notes)
        return slide

    # --- Type-specific renderers ---

    def _render_title(self, sd: SlideData):
        _, layout = self.layout_mapper.get_layout(sd)
        slide = self.prs.slides.add_slide(layout)
        self._set_placeholder_text(slide, 0, sd.title)
        # Subtitle from raw lines
        subtitle_parts = []
        for line in sd.raw_lines:
            if line.startswith('##'):
                subtitle_parts.append(line.lstrip('#').strip())
            elif line.startswith('#'):
                continue  # skip H1 duplicate
            else:
                subtitle_parts.append(line)
        if subtitle_parts:
            self._set_placeholder_text(slide, 1, '\n'.join(subtitle_parts))
        return slide

    def _render_section(self, sd: SlideData):
        _, layout = self.layout_mapper.get_layout(sd)
        slide = self.prs.slides.add_slide(layout)

        # Use the section name or slide title
        title_text = sd.title
        self._set_placeholder_text(slide, 0, title_text)

        # If there are bullets (e.g., section divider with numbered items), add to body
        if sd.bullets:
            body_ph = self._find_placeholder(slide, 1)
            if body_ph:
                self._add_bullets_to_textframe(body_ph.text_frame, sd.bullets)

        # If there are raw lines (e.g., italicized taglines), add to body
        if sd.raw_lines and not sd.bullets:
            body_ph = self._find_placeholder(slide, 1)
            if body_ph:
                tf = body_ph.text_frame
                tf.clear()
                for i, line in enumerate(sd.raw_lines):
                    para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    para.text = line
                    para.alignment = PP_ALIGN.CENTER

        return slide

    def _render_content(self, sd: SlideData):
        _, layout = self.layout_mapper.get_layout(sd)
        slide = self.prs.slides.add_slide(layout)

        # Title
        self._set_placeholder_text(slide, 0, sd.title)

        # Tables take priority for body content
        if sd.tables:
            self._add_table(slide, sd.tables[0], sd)
        elif sd.bullets:
            body_ph = self._find_placeholder(slide, 1)
            if body_ph:
                has_image = bool(sd.images)
                self._add_bullets_to_textframe(body_ph.text_frame, sd.bullets, has_image=has_image)

                # Append blockquotes as italic paragraphs
                if sd.blockquotes:
                    tf = body_ph.text_frame
                    for quote in sd.blockquotes:
                        para = tf.add_paragraph()
                        run = para.add_run()
                        run.text = quote
                        run.font.italic = True
                        run.font.size = Pt(12)
        elif sd.raw_lines:
            body_ph = self._find_placeholder(slide, 1)
            if body_ph:
                tf = body_ph.text_frame
                tf.clear()
                for i, line in enumerate(sd.raw_lines):
                    para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    segs = parse_rich_text(line)
                    self._add_rich_text_to_para(para, segs)

        # Image
        if sd.images:
            self._add_image(slide, sd.images[0])

        return slide

    def _render_closing(self, sd: SlideData):
        _, layout = self.layout_mapper.get_layout(sd)
        slide = self.prs.slides.add_slide(layout)
        self._set_placeholder_text(slide, 0, sd.title)
        return slide

    # --- Helper methods ---

    def _set_placeholder_text(self, slide, idx: int, text: str):
        ph = self._find_placeholder(slide, idx)
        if ph:
            ph.text = text

    def _find_placeholder(self, slide, idx: int):
        for shape in slide.placeholders:
            if shape.placeholder_format.idx == idx:
                return shape
        return None

    def _add_rich_text_to_para(self, para, segments: list[RichText]):
        """Add rich text segments to a paragraph, preserving formatting."""
        para.clear()
        for seg in segments:
            run = para.add_run()
            run.text = seg.text
            if seg.bold:
                run.font.bold = True
            if seg.italic:
                run.font.italic = True
            if seg.url:
                run.hyperlink.address = seg.url

    def _add_bullets_to_textframe(self, tf, bullets: list[BulletItem], has_image: bool = False):
        """Add bullet items to a text frame."""
        tf.clear()
        font_size = self._compute_font_size(len(bullets), has_image)

        for i, bullet in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.level = bullet.level
            self._add_rich_text_to_para(para, bullet.segments)
            for run in para.runs:
                run.font.size = font_size

    def _compute_font_size(self, bullet_count: int, has_image: bool) -> Pt:
        """Dynamic font sizing based on content density."""
        if has_image:
            thresholds = [(4, 14), (6, 12), (8, 11), (100, 10)]
        else:
            thresholds = [(5, 16), (7, 14), (10, 12), (100, 11)]
        for limit, size in thresholds:
            if bullet_count <= limit:
                return Pt(size)
        return Pt(10)

    def _add_table(self, slide, table: TableData, sd: SlideData):
        """Add a table shape to the slide."""
        total_rows = len(table.rows) + 1  # +1 for header
        col_count = table.col_count

        # Position: below title, spanning most of the slide width
        left = Inches(0.4)
        top = Inches(1.4)
        width = Inches(9.2)
        row_h = Inches(0.33) if total_rows <= 8 else Inches(0.27)
        height = row_h * total_rows

        tbl_shape = slide.shapes.add_table(total_rows, col_count, left, top, width, height)
        tbl = tbl_shape.table

        # Header row
        for j, header_segs in enumerate(table.headers):
            cell = tbl.cell(0, j)
            para = cell.text_frame.paragraphs[0]
            self._add_rich_text_to_para(para, header_segs)
            for run in para.runs:
                run.font.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY

        # Data rows
        for i, row_cells in enumerate(table.rows):
            for j, cell_segs in enumerate(row_cells):
                cell = tbl.cell(i + 1, j)
                para = cell.text_frame.paragraphs[0]
                self._add_rich_text_to_para(para, cell_segs)
                for run in para.runs:
                    run.font.size = Pt(9)

        # If there's also an image on this slide, add it
        if sd.images:
            # Place image below the table, smaller
            self._add_image(slide, sd.images[0], position="below_table",
                           top_offset=top + height + Inches(0.2))

    def _add_image(self, slide, image_path: str, position: str = "default",
                   top_offset=None):
        """Add an image to the slide, converting SVG to PNG if needed."""
        abs_path = self._resolve_path(image_path)
        if abs_path is None:
            return False

        # Convert SVG to PNG
        if abs_path.suffix.lower() == '.svg' and self.svg_converter:
            png_path = self.svg_converter.convert(abs_path)
            if png_path is None:
                if self.verbose:
                    print(f"    Warning: Could not convert SVG: {abs_path.name}")
                return False
            abs_path = png_path

        if not abs_path.exists():
            return False

        try:
            # Try PICTURE placeholder first (for content_image layout)
            for shape in slide.placeholders:
                if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type is not None:
                    # Check for PICTURE type (18)
                    try:
                        if shape.placeholder_format.type == 18:  # PP_PLACEHOLDER.PICTURE
                            shape.insert_picture(str(abs_path))
                            return True
                    except Exception:
                        continue

            # Fallback: add as freeform shape
            if position == "below_table":
                slide.shapes.add_picture(
                    str(abs_path),
                    Inches(2.0), top_offset or Inches(4.0),
                    width=Inches(6.0),
                )
            else:
                # Default: right side of slide
                slide.shapes.add_picture(
                    str(abs_path),
                    Inches(5.2), Inches(1.5),
                    width=Inches(4.3),
                )
            return True
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Could not insert image {abs_path.name}: {e}")
            return False

    def _resolve_path(self, image_path: str) -> Path | None:
        """Resolve a relative image path to absolute."""
        p = Path(image_path)
        if p.is_absolute():
            return p if p.exists() else None
        resolved = (self.base_dir / p).resolve()
        return resolved if resolved.exists() else None

    def _add_speaker_notes(self, slide, notes: str):
        """Add speaker notes to a slide."""
        if not notes:
            return
        try:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = notes
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Presentation Builder
# ---------------------------------------------------------------------------

class PresentationBuilder:
    """Orchestrate the full markdown-to-PPTX pipeline."""

    def __init__(self, template_path: Path, output_path: Path, input_path: Path,
                 skip_images: bool = False, verbose: bool = False):
        self.template_path = template_path
        self.output_path = output_path
        self.input_path = input_path
        self.base_dir = input_path.parent
        self.skip_images = skip_images
        self.verbose = verbose

    def build(self) -> Path:
        prs = Presentation(str(self.template_path))

        # Clear template sample slides
        self._clear_template_slides(prs)

        # Parse markdown
        md_text = self.input_path.read_text('utf-8')
        slides = parse_markdown_slides(md_text)

        if self.verbose:
            print(f"  Parsed {len(slides)} slides from {self.input_path.name}")
            type_counts = {}
            for s in slides:
                type_counts[s.slide_type] = type_counts.get(s.slide_type, 0) + 1
            print(f"  Types: {type_counts}")

        # SVG converter
        svg_converter = None
        if not self.skip_images:
            cache_dir = self.base_dir / "assets" / "generated" / ".png_cache"
            svg_converter = SVGConverter(cache_dir)
            if svg_converter._backend:
                if self.verbose:
                    print(f"  SVG backend: {svg_converter._backend}")
            else:
                print("  Warning: No SVG converter available. Images will be skipped.")

        # Layout mapper and renderer
        layout_mapper = LayoutMapper(prs)
        renderer = SlideRenderer(prs, layout_mapper, svg_converter, self.base_dir, self.verbose)

        # Render all slides
        for i, sd in enumerate(slides):
            if self.verbose:
                img_indicator = f" [{len(sd.images)} img]" if sd.images else ""
                tbl_indicator = f" [table]" if sd.tables else ""
                print(f"  [{i+1}/{len(slides)}] Slide {sd.slide_num}: "
                      f"{sd.slide_type}{img_indicator}{tbl_indicator} - {sd.title[:60]}")
            renderer.render(sd)

        # Save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(self.output_path))

        if self.verbose:
            print(f"\n  Saved: {self.output_path}")
            print(f"  Total slides: {len(prs.slides)}")

        return self.output_path

    def _clear_template_slides(self, prs: Presentation):
        """Remove existing sample slides from the template."""
        for i in range(len(prs.slides) - 1, -1, -1):
            rId = prs.slides._sldIdLst[i].rId
            prs.part.drop_rel(rId)
            del prs.slides._sldIdLst[i]


# ---------------------------------------------------------------------------
# Layout debug utility
# ---------------------------------------------------------------------------

def print_layout_debug(template_path: Path):
    """Print all layouts and their placeholders for debugging."""
    prs = Presentation(str(template_path))
    print(f"Template: {template_path.name}")
    print(f"Slide dimensions: {prs.slide_width / 914400:.1f}\" x {prs.slide_height / 914400:.1f}\"")
    print(f"Total layouts: {len(prs.slide_layouts)}")
    print(f"Existing slides: {len(prs.slides)}")
    print()

    for i, layout in enumerate(prs.slide_layouts):
        marker = ""
        for key, idx in LAYOUT_MAP.items():
            if idx == i:
                marker = f" <-- {key}"
        print(f"Layout {i}: {layout.name}{marker}")
        for ph in layout.placeholders:
            print(f"  ph[{ph.placeholder_format.idx}] type={ph.placeholder_format.type} "
                  f"name=\"{ph.name}\" "
                  f"size={ph.width/914400:.1f}\"x{ph.height/914400:.1f}\"")
        if not list(layout.placeholders):
            print("  (no placeholders)")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_template(arg_template: str | None) -> Path:
    """Find a template PPTX file."""
    if arg_template:
        p = Path(arg_template)
        if p.exists():
            return p
        print(f"Error: Template not found: {p}")
        sys.exit(1)

    for t in DEFAULT_TEMPLATES:
        if t.exists():
            return t

    print("Error: No template found. Use --template to specify one.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Markdown slides to PPTX (v2)"
    )
    parser.add_argument("--input", "-i", type=str, help="Markdown slides file")
    parser.add_argument("--template", "-t", type=str, default=None, help="PPTX template file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PPTX file")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="Skip SVG conversion and image embedding")
    parser.add_argument("--layout-debug", action="store_true", help="Print template layout info and exit")

    args = parser.parse_args()

    template_path = find_template(args.template)

    if args.layout_debug:
        print_layout_debug(template_path)
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / "presentation.pptx"

    print(f"Converting: {input_path.name}")
    print(f"Template:   {template_path.name}")

    builder = PresentationBuilder(
        template_path=template_path,
        output_path=output_path,
        input_path=input_path,
        skip_images=args.no_images,
        verbose=args.verbose,
    )

    result = builder.build()
    print(f"\nGenerated: {result}")


if __name__ == "__main__":
    main()
