"""Deterministic checks on the text a rendered slide shows.

A vision model reading screenshots caught defects the production reviewer passed
-- a leaked ``textttFlashSchNet`` on three slides, a raw ``[OpenReview](https://...)``
link, a source label ending on ``&`` -- at about 1.5 cents a slide. Every one of
those is a string property, so none of them needs a model: this reads the deck's
HTML and says so for free, before any screenshot is taken or any judge is paid.

It is static on purpose. The deck's HTML is a finished document, the browser only
switches slides, so the text a viewer reads is in the file. That makes the lint
fast, repeatable and testable without a browser, and lets the rendered-slide
inspector (which already runs before the paid vision pass) report its findings
next to its own.

Two severities:

* ``error`` -- text that is wrong on its face: a leaked LaTeX macro, a raw Markdown
  link or marker, a label cut off on a connector, a template placeholder.
* ``warning`` -- a style judgement: the figure repeats the slide's own title as its
  heading. Advice for a reviewer, never a block (it is common, and authoring-time
  lint in ``figures`` is where it is prevented).
"""

from __future__ import annotations

import html as html_lib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.display_text import TEXT_MACROS, ends_dangling

LINT_VERSION = "slide-text-lint/1"

_RAW_MACRO_RE = re.compile(r"\\(?:[A-Za-z]{2,})\s*\{")
_LEAKED_MACRO_NAME_RE = re.compile(
    r"(?<![A-Za-z\\])(?:" + "|".join(sorted(TEXT_MACROS, key=len, reverse=True)) + r")(?=[A-Z0-9])"
)
_RAW_MATH_RE = re.compile(r"\$[^$\n]{0,60}\\[A-Za-z]+[^$\n]{0,60}\$")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]\n]+\]\((?:https?://|mailto:|/|\.\./|#)[^)\s]*\)")
_MARKDOWN_PAIR_RE = re.compile(r"\*\*[^*\n]{1,80}\*\*|`[^`\n]{1,80}`|~~[^~\n]{1,80}~~")
# Only tokens that cannot be prose. "undefined behaviour" and NaN handling are
# ordinary subjects for an ML talk, so a bare word is a placeholder only when it is
# the WHOLE text block (a template that rendered nothing), never inside a sentence.
_PLACEHOLDER_RE = re.compile(r"\[object Object\]|\{\{[^}]*\}\}|\{%[^%]*%\}|lorem ipsum", re.I)
_NULL_BLOCK_RE = re.compile(r"^\s*(?:undefined|null|NaN|None)\s*$")


@dataclass
class SlideText:
    """What one slide shows, by role."""

    section_index: int
    slide_id: str = ""
    title: str = ""
    chip_labels: list[str] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    figure_texts: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return " ".join(html_lib.unescape(str(text or "")).split())


def _title_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _norm(text).lower()).strip()


_SVG_TEXT_RE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def svg_text_lines(svg_text: str) -> list[str]:
    """The text lines an SVG draws (each ``<text>`` element, tspans joined)."""

    lines = []
    for body in _SVG_TEXT_RE.findall(svg_text):
        line = _norm(_TAG_RE.sub("", body))
        if line:
            lines.append(line)
    return lines


class _DeckParser(HTMLParser):
    """Collect each ``<section>``'s title, chip labels, text and figure image paths."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.slides: list[SlideText] = []
        self.figure_paths: list[list[str]] = []
        self._section_depth = 0
        self._skip_depth = 0  # script/style/svg
        self._title_open = False
        self._chip_stack: list[dict[str, Any]] = []
        self._inline_svg: list[str] | None = None
        self._svg_buffer: list[str] = []

    @property
    def _current(self) -> SlideText | None:
        return self.slides[-1] if self._section_depth > 0 and self.slides else None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: (value or "") for name, value in attrs}
        classes = attributes.get("class", "").split()
        if tag == "section":
            if self._section_depth == 0:
                self.slides.append(SlideText(len(self.slides), slide_id=attributes.get("data-slide-id", "")))
                self.figure_paths.append([])
            self._section_depth += 1
            return
        current = self._current
        if current is None:
            return
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if tag == "svg":
            self._skip_depth += 1
            self._svg_buffer = []
            return
        if tag == "img" and attributes.get("src", "").lower().endswith(".svg"):
            self.figure_paths[-1].append(attributes["src"])
        if tag in {"h1", "h2"} and not current.title:
            self._title_open = True
        if "source-chip" in classes and "source-chip-poster" not in classes:
            if tag == "span" and "source-chip-dual" in classes:
                self._chip_stack.append({"tag": tag, "text": [], "dual": True, "done": False})
            elif tag in {"a", "span"} and not any(chip.get("dual") for chip in self._chip_stack):
                self._chip_stack.append({"tag": tag, "text": [], "dual": False, "done": False})
        elif tag == "a" and self._chip_stack and self._chip_stack[-1].get("dual") and "source-chip-poster" not in classes:
            if not self._chip_stack[-1]["done"]:
                self._chip_stack.append({"tag": "a", "text": [], "dual": False, "done": False, "child": True})

    def handle_endtag(self, tag: str) -> None:
        if tag == "section":
            if self._section_depth > 0:
                self._section_depth -= 1
            return
        if self._current is None and self._section_depth == 0:
            return
        if tag in {"script", "style"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "svg":
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in {"h1", "h2"}:
            self._title_open = False
        if self._chip_stack and self._chip_stack[-1]["tag"] == tag:
            chip = self._chip_stack.pop()
            label = _norm("".join(chip["text"]))
            if chip.get("child"):
                if self._chip_stack:
                    self._chip_stack[-1]["done"] = True
                if label and self._current is not None:
                    self._current.chip_labels.append(label)
            elif not chip["dual"] and label and self._current is not None:
                self._current.chip_labels.append(label)

    def handle_data(self, data: str) -> None:
        current = self._current
        if current is None or self._skip_depth:
            return
        text = _norm(data)
        if not text:
            return
        if self._title_open:
            current.title = (current.title + " " + text).strip()
        if self._chip_stack:
            self._chip_stack[-1]["text"].append(data)
        current.text_blocks.append(text)


def extract_slides(html_text: str, *, html_dir: Path | None = None) -> list[SlideText]:
    """Per-slide text from a deck's HTML; figure SVGs next to the HTML are read too."""

    parser = _DeckParser()
    parser.feed(html_text)
    parser.close()
    for slide, paths in zip(parser.slides, parser.figure_paths, strict=False):
        if html_dir is None:
            continue
        for relative in paths:
            candidate = (html_dir / relative).resolve()
            if candidate.is_file():
                slide.figure_texts.extend(svg_text_lines(candidate.read_text(encoding="utf-8", errors="replace")))
    return parser.slides


def _finding(code: str, severity: str, message: str, evidence: str) -> dict[str, str]:
    return {"check_id": code, "severity": severity, "message": message, "evidence": evidence[:160]}


def lint_slide(slide: SlideText, *, fragment_words: frozenset[str] = frozenset()) -> list[dict[str, str]]:
    """Every finding for one slide, errors first."""

    findings: list[dict[str, str]] = []
    visible = [slide.title, *slide.chip_labels, *slide.text_blocks, *slide.figure_texts]
    seen: set[tuple[str, str]] = set()

    def add(code: str, severity: str, message: str, evidence: str) -> None:
        key = (code, evidence)
        if key not in seen:
            seen.add(key)
            findings.append(_finding(code, severity, message, evidence))

    for text in visible:
        if not text:
            continue
        for pattern in (_RAW_MACRO_RE, _LEAKED_MACRO_NAME_RE, _RAW_MATH_RE):
            match = pattern.search(text)
            if match:
                add("leaked_latex_macro", "error",
                    "LaTeX markup or a leaked macro name is visible (a text macro such as \\texttt "
                    "must be unwrapped to its content)", text)
                break
        if _MARKDOWN_LINK_RE.search(text):
            add("raw_markdown_link", "error", "a raw Markdown link is visible; a slide shows the link's text only", text)
        elif _MARKDOWN_PAIR_RE.search(text):
            add("raw_markdown_marker", "error", "a raw Markdown emphasis or code marker is visible", text)
        if _PLACEHOLDER_RE.search(text) or _NULL_BLOCK_RE.match(text):
            add("template_placeholder", "error", "a template placeholder or null value is visible", text)

    for label in slide.chip_labels:
        if ends_dangling(label, fragment_words):
            add("dangling_label_end", "error",
                "a source label is cut off on a connector or function word", label)

    title_key = _title_key(slide.title)
    if title_key:
        for line in slide.figure_texts:
            if _title_key(line) == title_key:
                add("duplicate_title_in_figure", "warning",
                    "the figure repeats the slide's own title as its heading", line)
                break
    findings.sort(key=lambda item: (item["severity"] != "error", item["check_id"]))
    return findings


def lint_deck_html(
    html_path: Path,
    *,
    fragment_words: frozenset[str] = frozenset(),
) -> dict[int, list[dict[str, str]]]:
    """``{section_index: findings}`` for every slide that has any."""

    html_path = Path(html_path)
    slides = extract_slides(html_path.read_text(encoding="utf-8", errors="replace"), html_dir=html_path.parent)
    out: dict[int, list[dict[str, str]]] = {}
    for slide in slides:
        findings = lint_slide(slide, fragment_words=fragment_words)
        if findings:
            out[slide.section_index] = findings
    return out


def summarize(findings_by_section: dict[int, Sequence[dict[str, str]]]) -> dict[str, Any]:
    errors = sum(1 for items in findings_by_section.values() for f in items if f["severity"] == "error")
    warnings = sum(1 for items in findings_by_section.values() for f in items if f["severity"] == "warning")
    return {"lint_version": LINT_VERSION, "errors": errors, "warnings": warnings,
            "slides_with_findings": len(findings_by_section)}
