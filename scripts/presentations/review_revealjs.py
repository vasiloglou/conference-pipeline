#!/usr/bin/env python3
"""Review and fix formatting issues in a Reveal.js HTML presentation.

Goes through every slide in the generated HTML and checks/fixes:
- Aspect ratio: ensures slides will render at 16:9 (1280x720)
- Content overflow: detects slides with too many bullets or too-long text
- SVG sizing: ensures inline SVGs have proper viewBox and constraints
- Table formatting: checks tables aren't too wide
- Speaker notes: verifies notes are present
- Slide structure: validates each <section> has proper classes

Outputs a report and optionally writes a fixed HTML file.

Usage:
    python scripts/presentations/review_revealjs.py \\
        --input path/to/presentation.html \\
        --fix   # apply fixes in-place
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path


MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."


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


# ---------------------------------------------------------------------------
# Slide extraction
# ---------------------------------------------------------------------------

def extract_slides(html: str) -> list[dict]:
    """Extract individual slide sections from the HTML."""
    slides = []
    # Match top-level <section> elements (slides)
    pattern = r'<section([^>]*)>(.*?)</section>'
    for i, match in enumerate(re.finditer(pattern, html, re.DOTALL)):
        attrs = match.group(1)
        body = match.group(2)
        start = match.start()
        end = match.end()

        # Determine slide type from class
        class_match = re.search(r'class="([^"]*)"', attrs)
        slide_class = class_match.group(1) if class_match else ""

        if "slide-title" in slide_class:
            stype = "title"
        elif "slide-section" in slide_class:
            stype = "section"
        elif "slide-figure-zoom" in slide_class:
            stype = "figure-zoom"
        elif "slide-content-with-figure" in slide_class:
            stype = "content+figure"
        elif "slide-content" in slide_class:
            stype = "content"
        else:
            stype = "unknown"

        # Extract title from ### or # in markdown template
        title_match = re.search(r'###?\s+(.+?)(?:\n|$)', body)
        title = title_match.group(1).strip() if title_match else f"Slide {i+1}"

        # Count elements
        bullet_count = body.count('<li>') + len(re.findall(r'^- ', body, re.MULTILINE))
        has_table = bool(re.search(r'\|.+\|.+\|', body))
        svg_count = body.count('<div class="svg-container"')
        image_count = len(
            re.findall(r'<img\b[^>]*class="[^"]*\bslide-figure\b', body)
        )
        figure_count = svg_count + image_count
        has_notes = '<aside class="notes">' in body

        slides.append({
            "index": i + 1,
            "type": stype,
            "class": slide_class,
            "title": title[:60],
            "body": body,
            "attrs": attrs,
            "start": start,
            "end": end,
            "full_match": match.group(0),
            "bullet_count": bullet_count,
            "has_table": has_table,
            "svg_count": svg_count,
            "image_count": image_count,
            "figure_count": figure_count,
            "has_notes": has_notes,
        })

    return slides


# ---------------------------------------------------------------------------
# Issue checks
# ---------------------------------------------------------------------------

def check_slide(slide: dict) -> list[dict]:
    """Check a single slide for formatting issues. Returns list of issues."""
    issues = []
    stype = slide["type"]
    body = slide["body"]
    idx = slide["index"]

    # --- Bullet overflow ---
    if stype in ("content", "content+figure"):
        max_bullets = 8 if stype == "content" else 6
        if slide["bullet_count"] > max_bullets:
            issues.append({
                "slide": idx,
                "severity": "warn",
                "type": "bullet_overflow",
                "msg": f"{slide['bullet_count']} bullets (max recommended: {max_bullets})",
            })

    # --- Missing speaker notes ---
    if not slide["has_notes"] and stype != "title":
        issues.append({
            "slide": idx,
            "severity": "info",
            "type": "no_notes",
            "msg": "No speaker notes",
        })

    # --- SVG without viewBox ---
    svg_tags = re.findall(r'<svg[^>]*>', body)
    for svg_tag in svg_tags:
        if 'viewBox' not in svg_tag and 'viewbox' not in svg_tag:
            issues.append({
                "slide": idx,
                "severity": "warn",
                "type": "svg_no_viewbox",
                "msg": "SVG missing viewBox attribute — may not scale correctly",
            })
            break

    # --- SVG with excessive dimensions ---
    for svg_tag in svg_tags:
        w_match = re.search(r'width="(\d+)', svg_tag)
        h_match = re.search(r'height="(\d+)', svg_tag)
        if w_match and h_match:
            w, h = int(w_match.group(1)), int(h_match.group(1))
            if w > 1200 or h > 800:
                issues.append({
                    "slide": idx,
                    "severity": "warn",
                    "type": "svg_oversized",
                    "msg": f"SVG has large fixed dimensions ({w}x{h}px) — may overflow",
                })
                break

    # --- Content+figure with no rendered figure ---
    if stype == "content+figure" and slide["figure_count"] == 0:
        issues.append({
            "slide": idx,
            "severity": "error",
            "type": "figure_layout_no_figure",
            "msg": "Two-column layout but no figure found",
        })

    # --- Very long bullet text (>200 chars) ---
    long_bullets = re.findall(r'^- (.{200,})$', body, re.MULTILINE)
    if long_bullets:
        issues.append({
            "slide": idx,
            "severity": "info",
            "type": "long_bullet",
            "msg": f"{len(long_bullets)} bullet(s) exceed 200 chars — may wrap awkwardly",
        })

    # --- Table column count ---
    if slide["has_table"]:
        table_rows = re.findall(r'^\|(.+)\|$', body, re.MULTILINE)
        if table_rows:
            max_cols = max(row.count('|') + 1 for row in table_rows)
            if max_cols > 6:
                issues.append({
                    "slide": idx,
                    "severity": "warn",
                    "type": "wide_table",
                    "msg": f"Table has {max_cols} columns — may be too wide for slide",
                })

    # --- Unknown slide type ---
    if stype == "unknown":
        issues.append({
            "slide": idx,
            "severity": "error",
            "type": "unknown_type",
            "msg": f"Unrecognized slide class: '{slide['class']}'",
        })

    # --- Inline SVG font too small ---
    if stype in ("content+figure", "figure-zoom", "content", "section"):
        svg_blocks = re.findall(r'<svg[^>]*>.*?</svg>', body, re.DOTALL)
        for svg_block in svg_blocks:
            # Check XML attribute font sizes
            font_sizes = re.findall(r'font-size="([\d.]+)"', svg_block)
            # Check CSS inline font sizes
            font_sizes += re.findall(r'font-size:\s*([\d.]+)(?:px|pt)?', svg_block)
            small = [f for f in font_sizes if float(f) < 12]
            if small:
                smallest = min(float(f) for f in small)
                issues.append({
                    "slide": idx,
                    "severity": "warn",
                    "type": "inline_svg_font_too_small",
                    "msg": f"Inline SVG has {len(small)} font-size values below 12pt "
                           f"(smallest: {smallest:.1f}pt)",
                })
                break  # One warning per slide is enough

    # --- Multi-SVG vertical overflow ---
    if stype == "content+figure" and slide["svg_count"] >= 3:
        estimated_height = slide["svg_count"] * 230 + (slide["svg_count"] - 1) * 6
        issues.append({
            "slide": idx,
            "severity": "error",
            "type": "multi_svg_overflow",
            "msg": f"{slide['svg_count']} SVGs stacked vertically — "
                   f"estimated {estimated_height}px vs 520px container",
        })

    # --- Title text overflow in content+figure layout ---
    if stype == "content+figure":
        title_text = slide["title"]
        if len(title_text) > 58:
            issues.append({
                "slide": idx,
                "severity": "warn",
                "type": "title_text_overflow",
                "msg": f"Title '{title_text[:40]}...' ({len(title_text)} chars) — "
                       f"may clip at 55% max-width in content+figure layout",
            })

    # --- Long bullets in content+figure layout (stricter threshold) ---
    if stype == "content+figure":
        figure_long_bullets = re.findall(r'<li>(.{120,}?)</li>', body, re.DOTALL)
        if not figure_long_bullets:
            figure_long_bullets = re.findall(r'^- (.{120,})$', body, re.MULTILINE)
        if figure_long_bullets:
            issues.append({
                "slide": idx,
                "severity": "warn",
                "type": "long_bullet_figure_layout",
                "msg": f"{len(figure_long_bullets)} bullet(s) exceed 120 chars in "
                       f"content+figure layout — likely wraps into figure area",
            })

    # --- Graphviz transform scale collapse ---
    scale_matches = re.findall(r'class="graph"\s+transform="scale\(([\d.]+)', body)
    for scale_str in scale_matches:
        scale = float(scale_str)
        if scale < 0.5:
            font_matches = re.findall(r'font-size="([\d.]+)"', body)
            if font_matches:
                min_declared = min(float(f) for f in font_matches)
                effective = min_declared * scale
                if effective < 10:
                    issues.append({
                        "slide": idx,
                        "severity": "error",
                        "type": "graphviz_scale_collapse",
                        "msg": f"Graphviz SVG with scale={scale:.3f}, min font="
                               f"{min_declared:.0f}pt, effective={effective:.1f}pt — "
                               f"text will be unreadable",
                    })

    return issues


# ---------------------------------------------------------------------------
# Fixes
# ---------------------------------------------------------------------------

def apply_fixes(html: str, slides: list[dict], all_issues: list[dict]) -> str:
    """Apply automated fixes to the HTML. Returns modified HTML."""
    fixed = html

    # Fix 1: Add viewBox to SVGs that lack it (use width/height if available)
    def add_viewbox(match):
        tag = match.group(0)
        if 'viewBox' in tag or 'viewbox' in tag:
            return tag
        w_match = re.search(r'width="([^"]+)"', tag)
        h_match = re.search(r'height="([^"]+)"', tag)
        if w_match and h_match:
            w = w_match.group(1).replace('pt', '').replace('px', '').strip()
            h = h_match.group(1).replace('pt', '').replace('px', '').strip()
            try:
                wf, hf = float(w), float(h)
                tag = tag.replace('<svg', f'<svg viewBox="0 0 {wf} {hf}"', 1)
            except ValueError:
                pass
        return tag

    fixed = re.sub(r'<svg[^>]*>', add_viewbox, fixed)

    # Fix 2: Remove fixed width/height from SVGs so CSS can control sizing
    def normalize_svg_dims(match):
        tag = match.group(0)
        if 'viewBox' not in tag and 'viewbox' not in tag:
            return tag  # Skip if no viewBox (can't safely remove dims)
        # Remove absolute width/height so the CSS max-width/max-height take over
        tag = re.sub(r'\s+width="[^"]*"', '', tag)
        tag = re.sub(r'\s+height="[^"]*"', '', tag)
        return tag

    fixed = re.sub(r'<svg[^>]*>', normalize_svg_dims, fixed)

    # Fix 3: Bump inline SVG font sizes below 12pt
    def bump_inline_font_attr(match):
        val = float(match.group(1))
        if val < 12:
            return 'font-size="12"'
        return match.group(0)

    def bump_inline_font_css(match):
        prefix = match.group(1)
        val = float(match.group(2))
        unit = match.group(3) or "px"
        if val < 12:
            return f'{prefix}12{unit}'
        return match.group(0)

    # Only apply inside SVG blocks to avoid touching non-SVG HTML
    def fix_fonts_in_svg(svg_match):
        svg_text = svg_match.group(0)
        svg_text = re.sub(r'font-size="([\d.]+)"', bump_inline_font_attr, svg_text)
        svg_text = re.sub(r'(font-size:\s*)([\d.]+)(px|pt)?', bump_inline_font_css, svg_text)
        return svg_text

    fixed = re.sub(r'<svg[^>]*>.*?</svg>', fix_fonts_in_svg, fixed, flags=re.DOTALL)

    # Fix 4: Ensure Reveal.js config has proper 16:9 settings
    if 'width: 1280' not in fixed:
        fixed = fixed.replace(
            'Reveal.initialize({',
            'Reveal.initialize({\n    width: 1280,\n    height: 720,\n    margin: 0.04,'
        )

    return fixed


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(slides: list[dict], all_issues: list[dict]):
    """Print a formatted report of all slides and issues."""
    errors = [i for i in all_issues if i["severity"] == "error"]
    warnings = [i for i in all_issues if i["severity"] == "warn"]
    infos = [i for i in all_issues if i["severity"] == "info"]

    print(f"\n{'='*70}")
    print(f"SLIDE REVIEW REPORT")
    print(f"{'='*70}")
    print(f"Total slides: {len(slides)}")
    print(f"Issues found: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")

    # Slide type summary
    types = {}
    for s in slides:
        types[s["type"]] = types.get(s["type"], 0) + 1
    print(f"Slide types: {types}")

    # Notes coverage
    with_notes = sum(1 for s in slides if s["has_notes"])
    print(f"Speaker notes: {with_notes}/{len(slides)} slides")

    # Figure coverage: inline SVG figures and image figures both count.
    with_figures = sum(1 for s in slides if s["figure_count"] > 0)
    print(f"Slides with figures: {with_figures}/{len(slides)}")

    if all_issues:
        print(f"\n{'─'*70}")
        print("ISSUES BY SLIDE:")
        print(f"{'─'*70}")

        current_slide = None
        for issue in sorted(all_issues, key=lambda x: (x["slide"], x["severity"])):
            if issue["slide"] != current_slide:
                current_slide = issue["slide"]
                slide = slides[current_slide - 1]
                print(f"\n  Slide {current_slide}: {slide['title']}")
                print(f"    Type: {slide['type']}, Bullets: {slide['bullet_count']}, "
                      f"SVGs: {slide['svg_count']}, Images: {slide['image_count']}, Table: {slide['has_table']}")

            severity_icon = {"error": "!!!", "warn": " ! ", "info": " . "}
            icon = severity_icon.get(issue["severity"], "   ")
            print(f"    [{icon}] {issue['type']}: {issue['msg']}")

    print(f"\n{'='*70}")


def render_report(slides: list[dict], all_issues: list[dict]) -> str:
    """Render the existing formatted report as text for report-output artifacts."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_report(slides, all_issues)
    return buffer.getvalue()


def write_report_output(report_output: Path | None, slides: list[dict], all_issues: list[dict]) -> None:
    if report_output is None:
        return
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(render_report(slides, all_issues), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review and fix Reveal.js presentation")
    parser.add_argument(
        "--input",
        "-i",
        "--input-slides",
        dest="input_slides",
        type=Path,
        required=True,
        help="HTML presentation file",
    )
    parser.add_argument("--fix", action="store_true", help="Apply automated fixes in-place")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show errors and warnings")
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--topic", help="Selected topic metadata for manifest-backed runs.")
    parser.add_argument("--report-output", type=Path, help="Workspace-relative or absolute visual QC report output.")
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
            ("input_slides", "--input-slides"),
            ("report_output", "--report-output"),
        ):
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr == "report_output":
                _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dry_run": True,
        "input_slides": str(args.input_slides),
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "report_output": str(args.report_output),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.workspace is not None and args.dry_run:
        print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
        return 0

    input_path = args.input_slides
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    html = input_path.read_text('utf-8')
    print(f"Reviewing: {input_path.name} ({len(html)/1024:.0f} KB)")

    slides = extract_slides(html)
    if not slides:
        print("Error: No slides found in the HTML file")
        return 1

    # Check all slides
    all_issues = []
    for slide in slides:
        issues = check_slide(slide)
        all_issues.extend(issues)

    # Print report
    if args.quiet:
        important = [i for i in all_issues if i["severity"] in ("error", "warn")]
        if important:
            for issue in important:
                print(f"  Slide {issue['slide']}: [{issue['severity']}] {issue['type']}: {issue['msg']}")
        else:
            print("  No errors or warnings found.")
    else:
        print_report(slides, all_issues)

    write_report_output(args.report_output, slides, all_issues)

    # Apply fixes if requested
    if args.fix:
        print(f"\nApplying fixes...")
        fixed_html = apply_fixes(html, slides, all_issues)

        if fixed_html != html:
            input_path.write_text(fixed_html, encoding='utf-8')
            size_mb = input_path.stat().st_size / (1024 * 1024)
            print(f"Fixed file saved: {input_path} ({size_mb:.1f} MB)")
        else:
            print("No fixes needed.")

    # Return exit code based on errors
    error_count = sum(1 for i in all_issues if i["severity"] == "error")
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
