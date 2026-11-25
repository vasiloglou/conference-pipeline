#!/usr/bin/env python3
"""Validate SVG figures for presentation quality.

Scans SVG files for known failure patterns: font sizes below minimum,
transform scale collapse, invisible contrast, invalid attributes, and more.

Reports issues at ERROR/WARN/INFO severity levels.
Optionally auto-fixes simple issues (font sizes, viewBox, contrast).

Usage:
    python scripts/presentations/validate_svgs.py --topic 07
    python scripts/presentations/validate_svgs.py --topic 01,04,07
    python scripts/presentations/validate_svgs.py --all
    python scripts/presentations/validate_svgs.py --input path/to/file.svg
    python scripts/presentations/validate_svgs.py --topic 07 --fix
    python scripts/presentations/validate_svgs.py --topic 07 --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from conference_pipeline.presentation.svg_geometry import geometry_findings  # noqa: E402

# ---------------------------------------------------------------------------
# Project constants (same as sanity_check_slides.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."

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

# Midnight Executive palette
PALETTE = {
    "#1a1a2e", "#a8d8ea", "#ff6b6b", "#51cf66",
    "#ffd43b", "#ffd93d", "#9775fa", "#b088f9", "#4ecdc4",
    "#ffffff", "#e0e0e0", "#cccccc", "#888888", "#888",
    "#333344", "#2a2a4a", "#16213e",
    "none", "transparent", "white", "black",
}

# Font-size thresholds
MIN_FONT_ERROR = 12   # Below this → ERROR
MIN_FONT_WARN = 14    # Below this → WARN
MIN_EFFECTIVE_FONT = 10  # After transform scaling → ERROR

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    file: str
    check_id: str
    severity: str  # "error", "warn", "info"
    message: str
    line: int | None = None
    fixable: bool = False
    fixed: bool = False


@dataclass
class ValidationResult:
    files_scanned: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warn"]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "info"]


# ---------------------------------------------------------------------------
# SVG Validator
# ---------------------------------------------------------------------------

class SVGValidator:
    """Validates a single SVG file for known quality issues."""

    def __init__(self, path: Path, min_font: int = MIN_FONT_ERROR):
        self.path = path
        self.filename = path.name
        self.text = path.read_text("utf-8", errors="replace")
        self.min_font = min_font
        self.issues: list[Issue] = []

    def _add(self, check_id: str, severity: str, message: str,
             line: int | None = None, fixable: bool = False):
        self.issues.append(Issue(
            file=self.filename,
            check_id=check_id,
            severity=severity,
            message=message,
            line=line,
            fixable=fixable,
        ))

    # --- Parse helpers ---

    def _get_viewbox(self) -> tuple[float, float, float, float] | None:
        """Extract viewBox as (min_x, min_y, width, height)."""
        m = re.search(r'viewBox="([^"]+)"', self.text, re.IGNORECASE)
        if not m:
            return None
        parts = m.group(1).split()
        if len(parts) == 4:
            try:
                return tuple(float(p) for p in parts)
            except ValueError:
                return None
        return None

    def _get_root_dimensions(self) -> tuple[float | None, float | None]:
        """Extract width/height from the root <svg> tag."""
        svg_tag = re.search(r'<svg[^>]*>', self.text)
        if not svg_tag:
            return None, None
        tag = svg_tag.group(0)
        w = h = None
        wm = re.search(r'width="([^"]+)"', tag)
        hm = re.search(r'height="([^"]+)"', tag)
        if wm:
            w = _parse_dimension(wm.group(1))
        if hm:
            h = _parse_dimension(hm.group(1))
        return w, h

    def _get_scale_factor(self) -> float | None:
        """Extract graphviz-style transform scale from root <g> element."""
        m = re.search(
            r'class="graph"\s+transform="scale\(([\d.]+)\s*([\d.]+)?\)',
            self.text,
        )
        if m:
            return float(m.group(1))
        return None

    def _find_all_font_sizes(self) -> list[tuple[float, int | None]]:
        """Find all font-size values. Returns list of (size, line_number)."""
        results = []
        for i, line in enumerate(self.text.splitlines(), 1):
            # XML attribute: font-size="18.00"
            for m in re.finditer(r'font-size="([\d.]+)"', line):
                results.append((float(m.group(1)), i))
            # CSS inline: font-size: 12px  or  font-size:11px
            for m in re.finditer(r'font-size:\s*([\d.]+)(?:px|pt)?', line):
                val = float(m.group(1))
                # Skip if this is the same value as an XML attr on same line
                if (val, i) not in results:
                    results.append((val, i))
        return results

    # --- Checks ---

    def validate(self) -> list[Issue]:
        """Run all checks."""
        self._check_viewbox()
        self._check_font_sizes()
        self._check_transform_scale()
        self._check_contrast()
        self._check_marker_attributes()
        self._check_text_near_edge()
        self._check_oversized_canvas()
        self._check_thin_strokes()
        self._check_drawsvg_title_position()
        self._check_palette()
        self._check_file_size()
        self._check_geometry()
        return self.issues

    def _check_geometry(self):
        """Measured layout defects (labels crossing their box, dangling arrows,
        connectors through text, text off-canvas). Reported at WARN so the
        validation report documents them for review; the figure generator
        blocks on the same findings at codegen time and the vision QC stage
        remains the pixel-level authority."""
        try:
            findings = geometry_findings(self.text)
        except Exception as exc:  # pragma: no cover - never fail validation on the estimator
            self._add("geometry_check_error", "info", f"geometry check skipped: {exc}")
            return
        for finding in findings[:8]:
            self._add(
                f"geometry_{finding.check_id}",
                "warn" if finding.severity == "error" else "info",
                finding.message,
            )

    def _check_viewbox(self):
        """Check 1: SVG root must have viewBox."""
        vb = self._get_viewbox()
        if vb is None:
            w, h = self._get_root_dimensions()
            if w and h:
                self._add(
                    "missing_viewbox", "error",
                    f"No viewBox attribute (has width={w}, height={h} — can derive viewBox)",
                    fixable=True,
                )
            else:
                self._add(
                    "missing_viewbox", "error",
                    "No viewBox and no width/height — SVG will not scale correctly",
                )

    def _check_font_sizes(self):
        """Check 2: All font sizes must meet minimum threshold."""
        fonts = self._find_all_font_sizes()
        if not fonts:
            return

        scale = self._get_scale_factor()
        small_count = 0
        smallest = None

        for size, line_num in fonts:
            effective = size * scale if scale else size
            if effective < self.min_font:
                small_count += 1
                if smallest is None or effective < smallest:
                    smallest = effective

        if small_count > 0:
            extra = f" (scale={scale:.3f})" if scale else ""
            self._add(
                "font_too_small", "error",
                f"{small_count} font-size values below {self.min_font}pt{extra}, "
                f"smallest effective={smallest:.1f}pt",
                fixable=True,
            )

    def _check_transform_scale(self):
        """Check 3: Graphviz transform scale collapse."""
        scale = self._get_scale_factor()
        if scale is None or scale >= 0.5:
            return

        fonts = self._find_all_font_sizes()
        if not fonts:
            return

        min_declared = min(f[0] for f in fonts)
        effective = min_declared * scale

        if effective < MIN_EFFECTIVE_FONT:
            self._add(
                "transform_scale_collapse", "error",
                f"Graphviz scale={scale:.3f}, min declared font={min_declared:.0f}pt, "
                f"effective={effective:.1f}pt — text will be unreadable",
            )

    def _check_contrast(self):
        """Check 4: Black strokes/fills invisible on navy background."""
        # Skip content inside <defs>...</defs> (glyph definitions)
        text_no_defs = re.sub(r'<defs>.*?</defs>', '', self.text, flags=re.DOTALL)

        black_patterns = [
            (r'stroke[=:]\s*["\']?#000000', "stroke"),
            (r'stroke[=:]\s*["\']?#000["\'\s;]', "stroke"),
            (r'fill[=:]\s*["\']?#000000', "fill"),
            (r'fill[=:]\s*["\']?#000["\'\s;]', "fill"),
        ]

        found_types = set()
        for pattern, attr_type in black_patterns:
            if re.search(pattern, text_no_defs):
                found_types.add(attr_type)

        if found_types:
            attrs = " and ".join(sorted(found_types))
            self._add(
                "invisible_contrast", "error",
                f"Black {attrs} (#000/#000000) on navy background — will be invisible",
                fixable=True,
            )

    def _check_marker_attributes(self):
        """Check 5: Invalid marker-end/marker-start attributes."""
        if re.search(r'marker-end="True"', self.text, re.IGNORECASE):
            self._add(
                "invalid_marker_attribute", "error",
                'marker-end="True" is invalid SVG — arrowheads will not render',
                fixable=True,
            )
        if re.search(r'marker-start="True"', self.text, re.IGNORECASE):
            self._add(
                "invalid_marker_attribute", "error",
                'marker-start="True" is invalid SVG — arrowheads will not render',
                fixable=True,
            )

    def _check_text_near_edge(self):
        """Check 6: Text elements near viewBox edges (clipping risk)."""
        vb = self._get_viewbox()
        if not vb:
            return

        min_x, min_y, vb_w, vb_h = vb
        margin_x = vb_w * 0.05
        margin_y = vb_h * 0.05

        # Find text elements with x/y attributes
        text_positions = re.findall(
            r'<text[^>]*\bx="([^"]+)"[^>]*\by="([^"]+)"', self.text
        )
        text_positions += re.findall(
            r'<text[^>]*\by="([^"]+)"[^>]*\bx="([^"]+)"', self.text
        )

        near_edge_count = 0
        for x_str, y_str in text_positions:
            try:
                x, y = float(x_str), float(y_str)
            except ValueError:
                continue

            if (x < min_x + margin_x or x > min_x + vb_w - margin_x or
                    y < min_y + margin_y or y > min_y + vb_h - margin_y):
                near_edge_count += 1

        if near_edge_count > 0:
            self._add(
                "text_near_edge", "warn",
                f"{near_edge_count} text element(s) within 5% of viewBox edges — clipping risk",
            )

    def _check_oversized_canvas(self):
        """Check 7: Large canvas with scale transform."""
        vb = self._get_viewbox()
        scale = self._get_scale_factor()

        if vb and scale and scale < 0.8:
            _, _, vb_w, vb_h = vb
            if vb_w > 1000 or vb_h > 800:
                self._add(
                    "oversized_canvas", "warn",
                    f"Large viewBox ({vb_w:.0f}x{vb_h:.0f}) with scale={scale:.3f} — "
                    f"may render poorly at presentation scale",
                )

    def _check_thin_strokes(self):
        """Check 8: Stroke widths below 2pt."""
        # Find stroke-width values
        stroke_widths = re.findall(r'stroke-width[=:]\s*"?([\d.]+)', self.text)
        thin_count = sum(1 for sw in stroke_widths if float(sw) < 1.5)
        if thin_count > 5:  # Only flag if widespread
            self._add(
                "thin_strokes", "warn",
                f"{thin_count} elements with stroke-width < 1.5pt — may be invisible",
            )

    def _check_drawsvg_title_position(self):
        """Check 9: Center-origin SVG with title at bottom."""
        vb = self._get_viewbox()
        if not vb:
            return

        min_x, min_y, vb_w, vb_h = vb
        if min_x >= 0 and min_y >= 0:
            return  # Not center-origin

        # Find large text elements (likely titles, font-size >= 18)
        large_texts = []
        for m in re.finditer(
            r'<text[^>]*font-size="([\d.]+)"[^>]*\by="([^"]+)"', self.text
        ):
            fsize, y_str = float(m.group(1)), m.group(2)
            try:
                y = float(y_str)
            except ValueError:
                continue
            if fsize >= 18:
                large_texts.append(y)

        if not large_texts:
            return

        # In center-origin, negative Y is top. Title should be near min_y.
        canvas_mid_y = min_y + vb_h / 2
        topmost_title_y = min(large_texts)
        if topmost_title_y > canvas_mid_y:
            self._add(
                "drawsvg_title_position", "warn",
                f"Title text at y={topmost_title_y:.0f} is in bottom half of "
                f"center-origin canvas (viewBox min_y={min_y:.0f})",
            )

    def _check_palette(self):
        """Check 10: Colors outside Midnight Executive palette."""
        # Extract hex colors
        hex_colors = set(re.findall(r'[#]([0-9a-fA-F]{3,8})\b', self.text))
        normalized = set()
        for c in hex_colors:
            full = f"#{c.lower()}"
            # Normalize 3-char to 6-char
            if len(c) == 3:
                full = f"#{c[0]*2}{c[1]*2}{c[2]*2}"
            normalized.add(full)

        off_palette = normalized - {c.lower() for c in PALETTE if c.startswith("#")}
        # Filter out very common acceptable colors (grays, near-blacks, near-whites)
        truly_off = set()
        for c in off_palette:
            if len(c) != 7:
                continue
            try:
                r = int(c[1:3], 16)
                g = int(c[3:5], 16)
                b = int(c[5:7], 16)
            except ValueError:
                continue
            # Skip grays (r≈g≈b)
            if max(r, g, b) - min(r, g, b) < 30:
                continue
            truly_off.add(c)

        if len(truly_off) > 3:
            sample = ", ".join(sorted(truly_off)[:5])
            self._add(
                "off_palette_colors", "info",
                f"{len(truly_off)} non-palette colors used: {sample}",
            )

    def _check_file_size(self):
        """Check 11: Large SVG file."""
        size_kb = self.path.stat().st_size / 1024
        if size_kb > 500:
            self._add(
                "large_file_size", "info",
                f"SVG is {size_kb:.0f} KB — may slow HTML load",
            )


# ---------------------------------------------------------------------------
# Auto-fix functions
# ---------------------------------------------------------------------------

def fix_svg(path: Path, min_font: int = MIN_FONT_ERROR, backup: bool = True) -> list[Issue]:
    """Apply auto-fixes to an SVG file. Returns list of fixed issues."""
    text = path.read_text("utf-8", errors="replace")
    original = text
    fixed_issues = []

    # Fix 1: Add viewBox if missing
    svg_tag_match = re.search(r'<svg([^>]*)>', text)
    if svg_tag_match:
        tag = svg_tag_match.group(0)
        if 'viewBox' not in tag and 'viewbox' not in tag:
            w_m = re.search(r'width="([^"]+)"', tag)
            h_m = re.search(r'height="([^"]+)"', tag)
            if w_m and h_m:
                w = _parse_dimension(w_m.group(1))
                h = _parse_dimension(h_m.group(1))
                if w and h:
                    new_tag = tag.replace('<svg', f'<svg viewBox="0 0 {w} {h}"', 1)
                    text = text.replace(tag, new_tag, 1)
                    fixed_issues.append(Issue(
                        file=path.name, check_id="missing_viewbox",
                        severity="error", message=f"Added viewBox='0 0 {w} {h}'",
                        fixable=True, fixed=True,
                    ))

    # Fix 2: Bump small font sizes (XML attribute)
    def bump_font_attr(m):
        val = float(m.group(1))
        if val < min_font:
            return f'font-size="{min_font}"'
        return m.group(0)

    new_text = re.sub(r'font-size="([\d.]+)"', bump_font_attr, text)
    if new_text != text:
        count = sum(1 for a, b in zip(text, new_text) if a != b)  # rough diff
        fixed_issues.append(Issue(
            file=path.name, check_id="font_too_small",
            severity="error", message=f"Bumped small font-size attributes to {min_font}pt",
            fixable=True, fixed=True,
        ))
        text = new_text

    # Fix 2b: Bump small font sizes (CSS inline style)
    def bump_font_css(m):
        prefix = m.group(1)
        val = float(m.group(2))
        unit = m.group(3) or "px"
        if val < min_font:
            return f'{prefix}{min_font}{unit}'
        return m.group(0)

    new_text = re.sub(
        r'(font-size:\s*)([\d.]+)(px|pt)?',
        bump_font_css,
        text,
    )
    if new_text != text:
        fixed_issues.append(Issue(
            file=path.name, check_id="font_too_small",
            severity="error", message=f"Bumped small CSS font-size values to {min_font}",
            fixable=True, fixed=True,
        ))
        text = new_text

    # Fix 3: Replace black strokes/fills (outside <defs>)
    # Strategy: process the text in segments, skipping <defs>...</defs>
    def fix_contrast_outside_defs(svg_text: str) -> tuple[str, bool]:
        # Split around <defs>...</defs>
        parts = re.split(r'(<defs>.*?</defs>)', svg_text, flags=re.DOTALL)
        changed = False
        result = []
        for i, part in enumerate(parts):
            if part.startswith('<defs>'):
                result.append(part)  # Don't touch defs
            else:
                new_part = part
                # Replace black strokes with ice blue
                new_part = re.sub(r'stroke="#000000"', 'stroke="#a8d8ea"', new_part)
                new_part = re.sub(r'stroke="#000"', 'stroke="#a8d8ea"', new_part)
                new_part = re.sub(r'stroke:\s*#000000', 'stroke: #a8d8ea', new_part)
                new_part = re.sub(r'stroke:\s*#000([;\s"\'])', r'stroke: #a8d8ea\1', new_part)
                # Replace black fills on text elements with light gray
                # (be conservative: only replace fill on <text> elements)
                new_part = re.sub(
                    r'(<text[^>]*?)fill="#000000"',
                    r'\1fill="#e0e0e0"',
                    new_part,
                )
                new_part = re.sub(
                    r'(<text[^>]*?)fill="#000"',
                    r'\1fill="#e0e0e0"',
                    new_part,
                )
                if new_part != part:
                    changed = True
                result.append(new_part)
        return ''.join(result), changed

    new_text, contrast_changed = fix_contrast_outside_defs(text)
    if contrast_changed:
        fixed_issues.append(Issue(
            file=path.name, check_id="invisible_contrast",
            severity="error", message="Replaced black strokes/fills with visible colors",
            fixable=True, fixed=True,
        ))
        text = new_text

    # Fix 4: Remove invalid marker-end="True"
    if re.search(r'marker-end="True"', text, re.IGNORECASE):
        text = re.sub(r'\s*marker-end="True"', '', text, flags=re.IGNORECASE)
        fixed_issues.append(Issue(
            file=path.name, check_id="invalid_marker_attribute",
            severity="error", message="Removed invalid marker-end=\"True\" attribute",
            fixable=True, fixed=True,
        ))
    if re.search(r'marker-start="True"', text, re.IGNORECASE):
        text = re.sub(r'\s*marker-start="True"', '', text, flags=re.IGNORECASE)
        fixed_issues.append(Issue(
            file=path.name, check_id="invalid_marker_attribute",
            severity="error", message="Removed invalid marker-start=\"True\" attribute",
            fixable=True, fixed=True,
        ))

    # Write back if changed
    if text != original:
        if backup:
            shutil.copy2(path, path.with_suffix('.svg.bak'))
        path.write_text(text, encoding='utf-8')

    return fixed_issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dimension(val: str) -> float | None:
    """Parse a dimension string like '800', '1077.00pt', '500px'."""
    m = re.match(r'([\d.]+)', val.strip())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Topic / directory scanning
# ---------------------------------------------------------------------------

def validate_file(path: Path, min_font: int = MIN_FONT_ERROR) -> list[Issue]:
    """Validate a single SVG file."""
    v = SVGValidator(path, min_font=min_font)
    return v.validate()


def validate_directory(dir_path: Path, min_font: int = MIN_FONT_ERROR) -> ValidationResult:
    """Validate all SVGs in a directory."""
    result = ValidationResult()
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a directory", file=sys.stderr)
        return result

    svg_files = sorted(dir_path.rglob("*.svg"))
    result.files_scanned = len(svg_files)
    if not svg_files:
        result.issues.append(
            Issue(
                file=str(dir_path),
                check_id="no_svg_files",
                severity="error",
                message="No SVG files found; figure generation produced no SVG artifacts to validate",
            )
        )
        return result

    for svg in svg_files:
        issues = validate_file(svg, min_font=min_font)
        result.issues.extend(issues)

    return result


def validate_topic(topic_num: str, fix: bool = False, min_font: int = MIN_FONT_ERROR,
                   backup: bool = True) -> ValidationResult:
    """Validate all SVGs for a topic. Optionally auto-fix."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        print(f"Error: unknown topic '{topic_num}'", file=sys.stderr)
        return ValidationResult()

    gen_dir = PRESENTATIONS_DIR / topic_name / "assets" / "generated"
    if not gen_dir.is_dir():
        print(f"Warning: {gen_dir} not found", file=sys.stderr)
        return ValidationResult()

    # Auto-fix first if requested (before validation)
    fix_issues = []
    if fix:
        for svg in sorted(gen_dir.rglob("*.svg")):
            fix_issues.extend(fix_svg(svg, min_font=min_font, backup=backup))

    # Then validate (to show remaining issues)
    result = validate_directory(gen_dir, min_font=min_font)

    # Merge fix info
    for fi in fix_issues:
        result.issues.append(fi)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: ValidationResult, topic: str = ""):
    """Print human-readable report."""
    header = f"SVG VALIDATION REPORT"
    if topic:
        header += f": {topic}"

    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")
    print(f"Files scanned: {result.files_scanned}")
    print(f"Issues: {len(result.errors)} errors, {len(result.warnings)} warnings, "
          f"{len(result.infos)} info")

    # Fixed issues
    fixed = [i for i in result.issues if i.fixed]
    if fixed:
        print(f"Auto-fixed: {len(fixed)} issues")

    if result.errors:
        print(f"\n{'─'*70}")
        print("ERRORS:")
        print(f"{'─'*70}")
        _print_issues(result.errors)

    if result.warnings:
        print(f"\n{'─'*70}")
        print("WARNINGS:")
        print(f"{'─'*70}")
        _print_issues(result.warnings)

    if result.infos:
        print(f"\n{'─'*70}")
        print("INFO:")
        print(f"{'─'*70}")
        _print_issues(result.infos)

    print(f"\n{'='*70}")


def _print_issues(issues: list[Issue]):
    """Print grouped issues."""
    by_file: dict[str, list[Issue]] = {}
    for issue in issues:
        by_file.setdefault(issue.file, []).append(issue)

    for filename, file_issues in sorted(by_file.items()):
        print(f"\n  {filename}")
        for issue in file_issues:
            icon = {"error": "!!!", "warn": " ! ", "info": " . "}.get(issue.severity, "   ")
            fixed_tag = " [FIXED]" if issue.fixed else ""
            line_tag = f" (line {issue.line})" if issue.line else ""
            print(f"    [{icon}] {issue.check_id}: {issue.message}{line_tag}{fixed_tag}")


def print_json(result: ValidationResult, topic: str = ""):
    """Print JSON report."""
    data = validation_report_payload(result, topic=topic)
    print(json.dumps(data, indent=2))


def validation_report_payload(result: ValidationResult, topic: str = "") -> dict[str, object]:
    """Build the JSON-serializable validation report payload."""
    return {
        "topic": topic,
        "files_scanned": result.files_scanned,
        "summary": {
            "error": len(result.errors),
            "warn": len(result.warnings),
            "info": len(result.infos),
            "fixed": sum(1 for i in result.issues if i.fixed),
        },
        "issues": [asdict(i) for i in result.issues],
    }


def validation_report_text(result: ValidationResult, topic: str = "") -> str:
    """Build a compact text report for report-output artifacts."""
    lines = [
        "# SVG Validation Report",
        "",
        f"Topic: {topic or 'n/a'}",
        f"Files scanned: {result.files_scanned}",
        f"Errors: {len(result.errors)}",
        f"Warnings: {len(result.warnings)}",
        f"Info: {len(result.infos)}",
        "",
    ]
    if result.issues:
        lines.append("## Issues")
        lines.append("")
        for issue in result.issues:
            loc = f":{issue.line}" if issue.line is not None else ""
            fixed = " fixed" if issue.fixed else ""
            lines.append(
                f"- [{issue.severity.upper()}] {issue.file}{loc} {issue.check_id}{fixed}: {issue.message}"
            )
    else:
        lines.append("No issues found.")
    lines.append("")
    return "\n".join(lines)


def write_report_output(
    report_output: Path | None,
    result: ValidationResult,
    *,
    topic: str = "",
    json_output: bool = False,
) -> None:
    """Write the same validation report content to an explicit artifact path."""
    if report_output is None:
        return
    report_output.parent.mkdir(parents=True, exist_ok=True)
    if json_output:
        content = json.dumps(validation_report_payload(result, topic=topic), indent=2) + "\n"
    else:
        content = validation_report_text(result, topic=topic)
    report_output.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SVG figures for presentation quality",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Validate all topics")
    group.add_argument("--input", type=str, help="Single SVG file path")
    group.add_argument("--dir", type=str, help="Directory of SVG files")
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic number(s), comma-separated for legacy runs, or selected topic metadata for workspace runs",
    )

    parser.add_argument("--fix", action="store_true", help="Auto-fix simple issues")
    parser.add_argument("--no-backup", action="store_true", help="Skip .bak backup during fix")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--min-font", type=int, default=MIN_FONT_ERROR,
                        help=f"Minimum font size threshold (default: {MIN_FONT_ERROR})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all details including passing files")
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--report-output", type=Path, help="Workspace-relative or absolute validation report output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate workflow arguments and print planned JSON.")
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
                ("--report-output", args.report_output),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("input", "--input"),
            ("dir", "--dir"),
            ("report_output", "--report-output"),
        ):
            value = getattr(args, attr)
            if value is None:
                continue
            try:
                resolved, relative = _workspace_relative_path(Path(value), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr == "report_output":
                _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dir": str(args.dir) if args.dir is not None else None,
        "dry_run": True,
        "input": str(args.input) if args.input is not None else None,
        "json": args.json,
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "report_output": str(args.report_output),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.workspace is not None and args.dry_run:
        print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
        return 0

    backup = not args.no_backup

    total_result = ValidationResult()

    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(2)
        if args.fix:
            fix_issues = fix_svg(path, min_font=args.min_font, backup=backup)
            total_result.issues.extend(fix_issues)
        issues = validate_file(path, min_font=args.min_font)
        total_result.files_scanned = 1
        total_result.issues.extend(issues)

        if args.json:
            print_json(total_result, topic=path.name)
        else:
            print_report(total_result, topic=path.name)
        write_report_output(
            args.report_output,
            total_result,
            topic=path.name,
            json_output=args.json,
        )

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: {dir_path} is not a directory", file=sys.stderr)
            sys.exit(2)
        if args.fix:
            for svg in sorted(dir_path.rglob("*.svg")):
                fix_issues = fix_svg(svg, min_font=args.min_font, backup=backup)
                total_result.issues.extend(fix_issues)
        result = validate_directory(dir_path, min_font=args.min_font)
        total_result.files_scanned = result.files_scanned
        total_result.issues.extend(result.issues)

        if args.json:
            print_json(total_result, topic=str(dir_path))
        else:
            print_report(total_result, topic=str(dir_path))
        write_report_output(
            args.report_output,
            total_result,
            topic=str(dir_path),
            json_output=args.json,
        )

    elif args.topic or args.all:
        # Topic-based
        topics = list(TOPIC_NAMES.keys()) if args.all else args.topic.split(",")
        for t in topics:
            t = t.strip().zfill(2)
            if t not in TOPIC_NAMES:
                print(f"Warning: unknown topic '{t}', skipping", file=sys.stderr)
                continue

            result = validate_topic(t, fix=args.fix, min_font=args.min_font, backup=backup)
            total_result.files_scanned += result.files_scanned
            total_result.issues.extend(result.issues)

            if not args.json:
                print_report(result, topic=TOPIC_NAMES[t])

        if args.json:
            topic_label = "all" if args.all else args.topic
            print_json(total_result, topic=topic_label)
        write_report_output(
            args.report_output,
            total_result,
            topic="all" if args.all else args.topic,
            json_output=args.json,
        )

    else:
        build_arg_parser().error("one of --topic, --all, --input, or --dir is required")

    # Exit code
    error_count = len([i for i in total_result.issues
                       if i.severity == "error" and not i.fixed])
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
