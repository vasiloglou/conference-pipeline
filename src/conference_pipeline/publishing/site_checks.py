"""Checks over a built public site or hub, and the narrative sanitizer.

v6.0-Public-Release Phase 4 (spec sections 6.4, 6.5, 6.8, 6.9):

- ``check_links``: every ``href``/``src`` is a file in the site, an absolute ``https://``
  URL, a ``mailto:``-free ``#fragment``, never a network call;
- ``search_text``: the reference full-text ranking (BM25 over ``search-data.json``), the
  same formula the page's ``search.js`` runs in the browser;
- ``lint_accessibility``: an offline WCAG-oriented lint (spec D3);
- ``render_mobile``: a Playwright Chromium render at phone width, skipped with a named
  reason when Playwright or Chromium is absent, never passed silently;
- ``sanitize_html``: an allowlist filter for rendered narrative markdown.

Exit codes for the checks: 0 clean, 1 findings.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

BM25_K1 = 1.2
BM25_B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class CheckResult:
    exit_code: int
    findings: list[str] = field(default_factory=list)
    skipped: str | None = None


def _pages(site_dir: Path) -> list[Path]:
    return sorted(p for p in site_dir.rglob("*.html") if p.is_file())


def _own_pages(site_dir: Path) -> list[Path]:
    """The site's own pages; copied deck HTML is the bundle's and is checked as a file only."""
    return [p for p in _pages(site_dir) if not p.relative_to(site_dir).as_posix().startswith("files/")]


# --- links ------------------------------------------------------------------------


def check_links(site_dir: Path) -> CheckResult:
    findings: list[str] = []
    root = site_dir.resolve()
    for page in _own_pages(site_dir):
        doc = BeautifulSoup(page.read_text(encoding="utf-8"), "html.parser")
        for tag in doc.find_all(True):
            for attr in ("href", "src"):
                value = tag.get(attr)
                if not value:
                    continue
                parts = urlsplit(value)
                if parts.scheme == "https":
                    continue
                if parts.scheme or value.startswith("//"):
                    findings.append(f"{page.relative_to(site_dir)}: {attr}={value!r}: not https or relative")
                    continue
                if not parts.path:  # a #fragment on the same page
                    continue
                target = (page.parent / unquote(parts.path)).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    findings.append(f"{page.relative_to(site_dir)}: {attr}={value!r}: leaves the site")
                    continue
                if not target.is_file():
                    findings.append(f"{page.relative_to(site_dir)}: {attr}={value!r}: no such file")
    return CheckResult(1 if findings else 0, findings)


# --- search -----------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_rank(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Rank rows (each with ``text`` and optional ``title``) by BM25; zero-score rows dropped."""
    terms = tokenize(query)
    if not terms or not rows:
        return []
    docs = [tokenize(f"{row.get('title') or ''} {row.get('text') or ''}") for row in rows]
    avgdl = sum(len(d) for d in docs) / len(docs) or 1.0
    df = Counter(t for d in docs for t in set(d))
    scored = []
    for row, doc in zip(rows, docs):
        tf = Counter(doc)
        score = 0.0
        for term in terms:
            if tf[term]:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * (BM25_K1 + 1) / (tf[term] + BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avgdl))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: -item[0])
    return [row for _score, row in scored]


def search_text(site_dir: Path, query: str) -> list[dict[str, Any]]:
    """The site's ranked rows for ``query``, read the way the browser reads them."""
    if (site_dir / "search" / "meta.json").is_file():
        from conference_pipeline.publishing.search_shards import read_ranked

        return read_ranked(site_dir, query)
    rows = json.loads((site_dir / "search-data.json").read_text(encoding="utf-8"))
    return bm25_rank(rows, query)


# --- accessibility ----------------------------------------------------------------


def lint_accessibility(site_dir: Path) -> CheckResult:
    findings: list[str] = []
    for page in _own_pages(site_dir):
        name = page.relative_to(site_dir).as_posix()
        doc = BeautifulSoup(page.read_text(encoding="utf-8"), "html.parser")
        html_tag = doc.find("html")
        if html_tag is None or not html_tag.get("lang"):
            findings.append(f"{name}: <html> has no lang")
        if len(doc.find_all("h1")) != 1:
            findings.append(f"{name}: expected exactly one <h1>")
        levels = [int(h.name[1]) for h in doc.find_all(re.compile(r"^h[1-6]$"))]
        for before, after in zip(levels, levels[1:]):
            if after > before + 1:
                findings.append(f"{name}: heading level skips from h{before} to h{after}")
                break
        if doc.find("main") is None:
            findings.append(f"{name}: no <main> landmark")
        skip = doc.select_one("a[href^='#']")
        if skip is None or "skip" not in skip.get_text(" ").lower():
            findings.append(f"{name}: no skip link first")
        for img in doc.find_all("img"):
            if img.get("alt") is None:
                findings.append(f"{name}: <img> without alt")
        for control in doc.find_all(["input", "select", "textarea"]):
            if control.get("type") == "hidden":
                continue
            labelled = control.get("aria-label") or control.get("aria-labelledby") or (
                control.get("id") and doc.find("label", attrs={"for": control.get("id")})
            )
            if not labelled:
                findings.append(f"{name}: form control without an accessible name")
        for button in doc.find_all("button"):
            if not (button.get_text(strip=True) or button.get("aria-label")):
                findings.append(f"{name}: button without text")
        for media in doc.find_all(["video", "audio"]):
            if not media.has_attr("controls"):
                findings.append(f"{name}: <{media.name}> without controls")
        targets: dict[str, set[str]] = {}
        for a in doc.find_all("a", href=True):
            # The accessible name: aria-label wins over the text, as in the accname computation.
            text = a.get("aria-label") or " ".join(a.get_text(" ").split())
            targets.setdefault(text, set()).add(a["href"])
        for text, hrefs in targets.items():
            if len(hrefs) > 1:
                findings.append(f"{name}: {len(hrefs)} links share the text {text!r}")
    return CheckResult(1 if findings else 0, findings)


# --- mobile render ----------------------------------------------------------------

_OVERFLOW_JS = """
() => {
  const doc = document.documentElement;
  const clipped = [...document.querySelectorAll('h1,h2,h3')].filter(h => h.scrollWidth > h.clientWidth + 1).length;
  return {overflow: doc.scrollWidth > doc.clientWidth + 1, clipped};
}
"""


def render_mobile(site_dir: Path, *, width: int = 375, height: int = 812) -> CheckResult:
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ImportError:
        return CheckResult(0, [], skipped="playwright_not_installed")
    findings: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height})
            errors: list[str] = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            for path in _own_pages(site_dir):
                errors.clear()
                page.goto(path.resolve().as_uri())
                state = page.evaluate(_OVERFLOW_JS)
                name = path.relative_to(site_dir).as_posix()
                if state["overflow"]:
                    findings.append(f"{name}: horizontal overflow at {width}px")
                if state["clipped"]:
                    findings.append(f"{name}: {state['clipped']} clipped heading(s) at {width}px")
                findings.extend(f"{name}: page error: {err}" for err in errors)
            browser.close()
    except PlaywrightError as exc:
        return CheckResult(0, [], skipped=f"chromium_unavailable: {str(exc).splitlines()[0]}")
    return CheckResult(1 if findings else 0, findings)


# --- sanitizer --------------------------------------------------------------------

ALLOWED_TAGS = frozenset({
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "strong", "em", "code", "pre",
    "blockquote", "a", "table", "thead", "tbody", "tr", "th", "td", "hr", "br",
})
ALLOWED_ATTRS = {"a": {"href", "title", "aria-label"}, "th": {"align"}, "td": {"align"}}
_DROP_CONTENT = frozenset({"script", "style", "iframe", "object", "embed", "template"})


def _browser_href(value: str) -> str | None:
    """The href as a browser reads it, or None when it is not https, relative or a fragment.

    Browsers strip leading and trailing spaces and C0 controls, delete tabs and newlines
    anywhere, and read a backslash as a slash in http(s) URLs. So `` //host``, ``\\\\host``
    and ``/\\host`` are all protocol-relative (phase-4-r1 F9). The cleaned value is the one
    kept, so what is checked is what is emitted.
    """
    cleaned = "".join(c for c in value.strip("".join(map(chr, range(33)))) if c not in "\t\n\r")
    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        return None
    if cleaned.replace("\\", "/").startswith("//"):
        return None
    if urlsplit(cleaned).scheme not in ("", "https"):
        return None
    return cleaned


class _Sanitizer(HTMLParser):
    def __init__(self, heading_offset: int) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.drop_depth = 0
        self.heading_shift = heading_offset  # not .offset: HTMLParser uses that for its position

    def _tag(self, tag: str) -> str:
        if self.heading_shift and re.fullmatch(r"h[1-6]", tag):
            return f"h{min(6, int(tag[1]) + self.heading_shift)}"
        return tag

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_CONTENT:
            self.drop_depth += 1
            return
        if self.drop_depth or tag not in ALLOWED_TAGS:
            return
        kept = []
        for key, value in attrs:
            if key not in ALLOWED_ATTRS.get(tag, set()) or value is None:
                continue
            if key == "href":
                value = _browser_href(value)
                if value is None:
                    continue
            kept.append(f' {key}="{escape(value, quote=True)}"')
        self.out.append(f"<{self._tag(tag)}{''.join(kept)}>")

    def handle_endtag(self, tag):
        if tag in _DROP_CONTENT:
            self.drop_depth = max(0, self.drop_depth - 1)
            return
        if not self.drop_depth and tag in ALLOWED_TAGS and tag not in {"br", "hr"}:
            self.out.append(f"</{self._tag(tag)}>")

    def handle_data(self, data):
        if not self.drop_depth:
            self.out.append(escape(data, quote=False))


def sanitize_html(fragment: str, *, heading_offset: int = 0) -> str:
    """Allowlisted tags and attributes only; ``href`` limited to https, relative and ``#``."""
    parser = _Sanitizer(heading_offset)
    parser.feed(fragment)
    parser.close()
    return "".join(parser.out)


_LINK_OR_EM = re.compile(r'<em>(?P<em>.*?)</em>|<a href="(?P<href>[^"]*)">(?P<text>.*?)</a>', re.S)
_TAG = re.compile(r"<[^>]+>")


def _name_repeated_links(html: str, reserved: Mapping[str, str] | None = None) -> str:
    """Links that share a text but go to different places get an ``aria-label``: the text and
    the nearest emphasised title before the link (a narrative's ``*Title* — id, [OpenReview](...)``),
    else the target. A screen reader's links list then tells them apart (6b, ICML site check)."""
    # ``reserved``: link texts the page itself uses (its site name -> home page), so a narrative
    # link reusing one with another target is named apart too (6b, NeurIPS 2025).
    targets: dict[str, set[str]] = {escape(text, quote=False): {href} for text, href in (reserved or {}).items()}
    for m in _LINK_OR_EM.finditer(html):
        if m.group("href") is not None:
            targets.setdefault(m.group("text"), set()).add(m.group("href"))
    repeated = {text for text, hrefs in targets.items() if len(hrefs) > 1}
    if not repeated:
        return html
    last_em: str | None = None

    def replace(m: re.Match[str]) -> str:
        nonlocal last_em
        if m.group("em") is not None:
            last_em = " ".join(unescape(_TAG.sub("", m.group("em"))).split())
            return m.group(0)
        text, href, title = m.group("text"), m.group("href"), last_em
        last_em = None
        if text not in repeated:
            return m.group(0)
        label = f"{unescape(_TAG.sub('', text)).strip()}: {title or unescape(href)}"
        return f'<a href="{href}" aria-label="{escape(label, quote=True)}">{text}</a>'

    return _LINK_OR_EM.sub(replace, html)


_HEADING = re.compile(r"<(/?)h([1-6])>")


def _level_headings(html: str, floor: int) -> str:
    """No heading more than one level below the one before it (the page's own sits at
    ``floor``): a narrative's #### after ## would skip a level (6b, NeurIPS 2025)."""
    previous = floor
    open_levels: list[int] = []

    def replace(match: re.Match[str]) -> str:
        nonlocal previous
        if match.group(1):
            return f"</h{open_levels.pop() if open_levels else match.group(2)}>"
        level = min(int(match.group(2)), previous + 1, 6)
        previous = level
        open_levels.append(level)
        return f"<h{level}>"

    return _HEADING.sub(replace, html)


def render_markdown(text: str, *, heading_offset: int = 1, reserved: Mapping[str, str] | None = None) -> str:
    """A narrative as sanitized HTML, its headings shifted under the page's own <h1>, with no
    level skipped, and links that share a text named apart."""
    import markdown

    html = sanitize_html(markdown.markdown(text, extensions=["tables"]), heading_offset=heading_offset)
    return _name_repeated_links(_level_headings(html, heading_offset + 1), reserved)
