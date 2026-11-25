"""Screenshot-backed rendered slide quality checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from conference_pipeline.workflow_runs import write_json_atomic

SCHEMA_VERSION = 1
DEFAULT_VIEWPORT = (1280, 720)
_NONBLANK_PIXEL_THRESHOLD = 0.003
# A few px of overshoot is normal (descenders, sub-pixel layout, rounded line
# boxes); a cropped list runs tens of px past the edge.
_TEXT_OVERFLOW_PX_THRESHOLD = 8.0
# Roughly a 20x20px patch. Adjacent line boxes can share a pixel row or two
# without anything being wrong; text landing on text covers far more.
_TEXT_OVERLAP_AREA_THRESHOLD = 400.0


def inspect_rendered_slides(
    input_html: Path,
    *,
    output_report: Path,
    screenshots_dir: Path,
    viewport: tuple[int, int] = DEFAULT_VIEWPORT,
) -> dict[str, object]:
    """Render a local HTML deck and emit screenshot-backed QC evidence."""

    html_path = input_html.expanduser().resolve()
    if not html_path.is_file():
        raise FileNotFoundError(f"input HTML does not exist: {html_path}")
    workspace = _infer_workspace(html_path)
    report_path = _resolve_like(output_report, base=html_path.parent)
    screenshot_root = _resolve_like(screenshots_dir, base=html_path.parent)
    width, height = viewport
    if width <= 0 or height <= 0:
        raise ValueError("viewport dimensions must be positive")

    screenshot_root.mkdir(parents=True, exist_ok=True)
    _clear_stale_slide_screenshots(screenshot_root)
    slides: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    # What the slides SAY, checked from the deck's own HTML: a leaked LaTeX macro,
    # a raw Markdown link, a label cut off on a connector. These are string
    # properties, so they are found here for free, before the paid vision pass.
    text_lint = _lint_deck_text(html_path)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment dependent
        report = _blocker_report(
            html_path,
            workspace,
            width,
            height,
            f"playwright import failed: {exc}",
        )
        write_json_atomic(report_path, report)
        return report

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:  # pragma: no cover - environment dependent
            report = _blocker_report(
                html_path,
                workspace,
                width,
                height,
                f"playwright chromium launch failed: {exc}",
            )
            write_json_atomic(report_path, report)
            return report
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(f"{html_path.as_uri()}?hidecontrols", wait_until="networkidle")
            sections = page.locator("section")
            section_count = sections.count()
            slide_indices = _mapped_slide_indices(html_path.parent, section_count)
            if section_count == 0:
                sections = page.locator("body")
                slide_indices = [0]
            for slide_number, section_index in enumerate(slide_indices, start=1):
                page.evaluate(_JS_SHOW_SLIDE, section_index)
                locator = sections.nth(section_index)
                screenshot_path = screenshot_root / f"slide_{slide_number:03d}.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                metrics = locator.evaluate(_JS_METRICS)
                pixel_ratio = _nonblank_pixel_ratio(screenshot_path)
                slide_errors = _errors_from_metrics(
                    slide_number=slide_number,
                    screenshot_path=screenshot_path,
                    workspace=workspace,
                    metrics=metrics,
                    pixel_ratio=pixel_ratio,
                )
                text_findings = text_lint.get(section_index, [])
                for finding in text_findings:
                    row = _text_lint_row(slide_number, finding, _relative(screenshot_path, workspace))
                    (slide_errors if finding["severity"] == "error" else warnings).append(row)
                errors.extend(slide_errors)
                slides.append(
                    {
                        "slide_number": slide_number,
                        "section_index": section_index,
                        "screenshot_path": _relative(screenshot_path, workspace),
                        "pixel_nonblank_ratio": pixel_ratio,
                        "metrics": metrics,
                        "errors": slide_errors,
                        "text_lint": text_findings,
                    }
                )
        finally:
            browser.close()

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if errors else "passed",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_html": _relative(html_path, workspace),
        "viewport": {"width": width, "height": height},
        "screenshots_dir": _relative(screenshot_root, workspace),
        "slides": slides,
        "errors": errors,
        "warnings": warnings,
        "text_lint": _summarize_text_lint(text_lint),
    }
    write_json_atomic(report_path, report)
    return report


def _blocker_report(
    html_path: Path,
    workspace: Path,
    width: int,
    height: int,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_html": _relative(html_path, workspace),
        "viewport": {"width": width, "height": height},
        "screenshots_dir": None,
        "slides": [],
        "errors": [
            {
                "slide_number": None,
                "check_id": "playwright_available",
                "severity": "error",
                "message": reason,
                "screenshot_path": None,
            }
        ],
    }


def _infer_workspace(path: Path) -> Path:
    parts = path.resolve().parts
    if "decks" in parts:
        index = parts.index("decks")
        return Path(*parts[:index])
    return path.parent


def _resolve_like(path: Path, *, base: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _mapped_slide_indices(deck_dir: Path, section_count: int) -> list[int]:
    # Reveal.js decks can include generated figure-zoom sections that are not
    # listed in presentation_slide_mapping.json. Those sections still render in
    # PDFs and videos, so screenshot QC must inspect every DOM section rather
    # than only the mapped content slides.
    all_indices = list(range(section_count))
    mapping_path = deck_dir / "presentation_slide_mapping.json"
    if not mapping_path.is_file():
        return all_indices
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return all_indices
    if not isinstance(payload, dict):
        return all_indices

    indices: list[int] = []
    seen: set[int] = set()
    for key in sorted(
        payload,
        key=lambda item: (0, int(item)) if str(item).isdigit() else (1, str(item)),
    ):
        value = payload.get(key)
        if not isinstance(value, int) or value < 0 or value >= section_count or value in seen:
            continue
        indices.append(value)
        seen.add(value)
    if not indices:
        return all_indices
    return indices + [index for index in all_indices if index not in seen]


def _clear_stale_slide_screenshots(root: Path) -> None:
    """Remove screenshots from a previous render before writing current evidence."""

    for path in root.glob("slide_*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            path.unlink()


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _nonblank_pixel_ratio(path: Path) -> float:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((320, 180))
        pixels = list(image.getdata())
    if not pixels:
        return 0.0
    nonblank = sum(1 for red, green, blue in pixels if min(red, green, blue) < 245)
    return nonblank / len(pixels)


def _lint_deck_text(html_path: Path) -> dict[int, list[dict[str, str]]]:
    # Imported here: structured_video is a heavy module and this one is not.
    from conference_pipeline.presentation.slide_text_lint import lint_deck_html
    from conference_pipeline.presentation.structured_video import SVG_FRAGMENT_END_WORDS

    return lint_deck_html(html_path, fragment_words=SVG_FRAGMENT_END_WORDS)


def _summarize_text_lint(findings: dict[int, list[dict[str, str]]]) -> dict[str, object]:
    from conference_pipeline.presentation.slide_text_lint import summarize

    return summarize(findings)


def _text_lint_row(slide_number: int, finding: dict[str, str], screenshot_rel: str) -> dict[str, object]:
    row = _error(slide_number, finding["check_id"], f"{finding['message']}: {finding['evidence']!r}", screenshot_rel)
    row["severity"] = finding["severity"]
    row["source"] = "slide_text_lint"
    return row


def _errors_from_metrics(
    *,
    slide_number: int,
    screenshot_path: Path,
    workspace: Path,
    metrics: dict[str, Any],
    pixel_ratio: float,
) -> list[dict[str, object]]:
    screenshot_rel = _relative(screenshot_path, workspace)
    errors: list[dict[str, object]] = []
    if metrics.get("document_title_slide_count", 0) == 0:
        errors.append(_error(slide_number, "title slide", "title slide is missing", screenshot_rel))
    if metrics.get("document_title_logo_count", 0) == 0:
        errors.append(_error(slide_number, "title logo", "title logo is missing", screenshot_rel))
    if metrics.get("document_decorative_art_count", 0) == 0:
        errors.append(
            _error(slide_number, "decorative art", "decorative art is missing", screenshot_rel)
        )
    if metrics.get("visible_file_path_text"):
        errors.append(
            _error(
                slide_number,
                "file-path text",
                "visible file-path text appears in rendered slide",
                screenshot_rel,
            )
        )
    if not metrics.get("is_title_slide"):
        if not metrics.get("section_has_content_brand_bug"):
            errors.append(
                _error(
                    slide_number,
                    "content brand bug",
                    "content brand bug is missing",
                    screenshot_rel,
                )
            )
        if not metrics.get("section_has_source_chips") and not metrics.get(
            "section_chips_not_applicable"
        ):
            # A slide may declare data-qc-chips="not-applicable" when it has no
            # cited sources to chip -- the executive itinerary index, whose rows
            # are the conference's own topics and whose sources live on the
            # closing resources slide. The declaration is in the rendered HTML,
            # so it is auditable; absence of chips without it is still an error.
            errors.append(
                _error(slide_number, "source chips", "source chips are missing", screenshot_rel)
            )
    for image in metrics.get("images", []):
        if not image.get("loaded"):
            errors.append(
                _error(slide_number, "asset_loaded", "image asset did not load", screenshot_rel)
            )
        if image.get("is_slide_visual") and image.get("clipped"):
            errors.append(
                _error(
                    slide_number,
                    "figure_clipped",
                    "main slide visual extends beyond the slide canvas",
                    screenshot_rel,
                )
            )
    for svg in metrics.get("svgs", []):
        if int(svg.get("shape_count", 0)) == 0:
            errors.append(
                _error(slide_number, "svg_nonblank", "SVG has no visible child shapes", screenshot_rel)
            )
        if svg.get("clipped"):
            errors.append(
                _error(
                    slide_number,
                    "svg_not_clipped",
                    "SVG bounding box is clipped; cropped logos are render blockers",
                    screenshot_rel,
                )
            )
    if metrics.get("text_figure_overlap"):
        errors.append(
            _error(slide_number, "no_text_figure_overlap", "text overlaps a figure", screenshot_rel)
        )
    text_overflow = float(metrics.get("text_overflow_px") or 0.0)
    if text_overflow > _TEXT_OVERFLOW_PX_THRESHOLD:
        errors.append(
            _error(
                slide_number,
                "text_within_slide",
                f"text extends {text_overflow:.0f}px beyond the slide canvas and is "
                "cropped by the slide's overflow rule",
                screenshot_rel,
            )
        )
    text_overlap = float(metrics.get("text_text_overlap_px") or 0.0)
    if text_overlap > _TEXT_OVERLAP_AREA_THRESHOLD:
        errors.append(
            _error(
                slide_number,
                "no_text_text_overlap",
                f"text is painted over other text ({text_overlap:.0f}px^2 of overlap)",
                screenshot_rel,
            )
        )
    if pixel_ratio < _NONBLANK_PIXEL_THRESHOLD:
        errors.append(_error(slide_number, "slide_not_blank", "slide renders blank", screenshot_rel))
    if metrics.get("tiny_text_count", 0) > 0:
        errors.append(
            _error(slide_number, "text_readable", "text below readable size threshold", screenshot_rel)
        )
    return errors


def _error(
    slide_number: int,
    check_id: str,
    message: str,
    screenshot_path: str,
) -> dict[str, object]:
    return {
        "slide_number": slide_number,
        "check_id": check_id,
        "severity": "error",
        "message": message,
        "screenshot_path": screenshot_path,
    }


_JS_METRICS = """
(section) => {
  const rectOf = (el) => {
    const rect = el.getBoundingClientRect();
    return {left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom,
            width: rect.width, height: rect.height};
  };
  const rectFromClientRect = (rect) => ({
    left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom,
    width: rect.width, height: rect.height
  });
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      Number(style.opacity || 1) > 0 && rect.width > 1 && rect.height > 1;
  };
  const sectionRect = section.getBoundingClientRect();
  const documentTitleSlides = Array.from(document.querySelectorAll(
    'section.slide-title, section[data-slide-type="title"]'
  ));
  const documentTitleLogos = Array.from(document.querySelectorAll(
    '.title-logo, [data-qc-role="title-logo"]'
  ));
  const documentDecorativeArt = Array.from(document.querySelectorAll(
    '.decorative-art, [data-qc-role="decorative-art"]'
  ));
  const svgs = Array.from(section.querySelectorAll('svg')).filter(visible).map((svg) => {
    const shapes = Array.from(svg.querySelectorAll(
      'path,circle,rect,line,polyline,polygon,text,image,ellipse'
    )).filter(visible);
    const rect = svg.getBoundingClientRect();
    return {
      shape_count: shapes.length,
      clipped: rect.left < sectionRect.left - 2 || rect.top < sectionRect.top - 2 ||
        rect.right > sectionRect.right + 2 || rect.bottom > sectionRect.bottom + 2,
      bounds: rectOf(svg)
    };
  });
  const images = Array.from(section.querySelectorAll('img')).filter(visible).map((img) => {
    const rect = img.getBoundingClientRect();
    return {
      loaded: Boolean(img.complete && img.naturalWidth > 0 && img.naturalHeight > 0),
      is_slide_visual: img.classList.contains('slide-visual'),
      clipped: rect.left < sectionRect.left - 2 || rect.top < sectionRect.top - 2 ||
        rect.right > sectionRect.right + 2 || rect.bottom > sectionRect.bottom + 2,
      bounds: rectOf(img)
    };
  });
  const figureElements = [...Array.from(section.querySelectorAll('svg,img'))].filter(visible);
  const figureRects = figureElements.map(rectOf);
  const textElements = Array.from(section.querySelectorAll('h1,h2,h3,h4,h5,h6,p,li,span,strong,em,text'))
    .filter((el) => visible(el) && (el.textContent || '').trim().length > 0 && !el.closest('svg'));
  const textRects = textElements.flatMap((el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    return Array.from(range.getClientRects())
      .filter((rect) => rect.width > 1 && rect.height > 1)
      .map(rectFromClientRect);
  });
  const overlap = (a, b) => {
    const width = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
    const height = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    return width * height;
  };
  const textFigureOverlap = textRects.some((textRect) =>
    figureRects.some((figureRect) => overlap(textRect, figureRect) > 5000)
  );
  // Text running past the slide edge, and text painted over other text. Both
  // are invisible to the checks above (which only measure svg/img clipping and
  // text-vs-figure overlap), and both are exactly what an over-filled list
  // produces: section { overflow: hidden } silently crops the tail, and an
  // absolutely-positioned footer lands on top of whatever grew into it.
  const BLOCK_SELECTOR = 'h1,h2,h3,h4,h5,h6,p,li';
  const textBlocks = Array.from(section.querySelectorAll(BLOCK_SELECTOR))
    .filter((el) => visible(el) && (el.textContent || '').trim().length > 0 &&
      !el.closest('svg') && !el.querySelector(BLOCK_SELECTOR));
  const lineRectsOf = (el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    return Array.from(range.getClientRects())
      .filter((rect) => rect.width > 1 && rect.height > 1)
      .map(rectFromClientRect);
  };
  const blockLines = textBlocks.map((el) => lineRectsOf(el));
  let textOverflowPx = 0;
  blockLines.forEach((lines) => {
    lines.forEach((rect) => {
      textOverflowPx = Math.max(
        textOverflowPx,
        rect.bottom - sectionRect.bottom,
        sectionRect.top - rect.top,
        rect.right - sectionRect.right,
        sectionRect.left - rect.left
      );
    });
  });
  let textTextOverlapPx = 0;
  for (let i = 0; i < blockLines.length; i += 1) {
    for (let j = i + 1; j < blockLines.length; j += 1) {
      blockLines[i].forEach((a) => {
        blockLines[j].forEach((b) => {
          textTextOverlapPx = Math.max(textTextOverlapPx, overlap(a, b));
        });
      });
    }
  }
  const tinyTextCount = textElements.filter((el) =>
    parseFloat(window.getComputedStyle(el).fontSize || '0') < 10
  ).length;
  const visibleText = (section.innerText || '').trim();
  const visibleFilePathText = /\\b(?:decks\\/|assets\\/)[^\\s<]+?\\.(?:svg|png|jpe?g|webp)\\b/.test(visibleText);
  return {
    section_text_length: visibleText.length,
    is_title_slide: section.classList.contains('slide-title') ||
      section.getAttribute('data-slide-type') === 'title',
    document_title_slide_count: documentTitleSlides.length,
    document_title_logo_count: documentTitleLogos.length,
    document_decorative_art_count: documentDecorativeArt.length,
    section_has_content_brand_bug: Boolean(section.querySelector(
      '.content-brand-bug, [data-qc-role="content-brand-bug"]'
    )),
    section_has_source_chips: Boolean(section.querySelector(
      '.reference-chips, [data-qc-role="source-chips"], .source-chip'
    )),
    section_chips_not_applicable:
      section.getAttribute('data-qc-chips') === 'not-applicable',
    visible_file_path_text: visibleFilePathText,
    svg_count: svgs.length,
    image_count: images.length,
    svgs,
    images,
    text_figure_overlap: textFigureOverlap,
    text_overflow_px: textOverflowPx,
    text_text_overlap_px: textTextOverlapPx,
    tiny_text_count: tinyTextCount
  };
}
"""

_JS_SHOW_SLIDE = """
(index) => {
  const sections = Array.from(document.querySelectorAll('section'));
  if (sections.length === 0) {
    return;
  }
  document.body.style.margin = '0';
  document.body.style.overflow = 'hidden';
  sections.forEach((section, current) => {
    section.classList.toggle('present', current === index);
    section.classList.toggle('past', current < index);
    section.classList.toggle('future', current > index);
    section.style.display = current === index ? 'block' : 'none';
    section.style.visibility = current === index ? 'visible' : 'hidden';
    section.style.opacity = current === index ? '1' : '0';
    section.style.position = current === index ? 'relative' : 'absolute';
    section.style.left = '0';
    section.style.top = '0';
    section.style.width = '1280px';
    section.style.height = '720px';
    section.style.transform = 'none';
  });
}
"""
