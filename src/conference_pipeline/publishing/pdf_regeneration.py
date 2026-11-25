"""Deck PDFs regenerated from the deck HTML the release publishes (v6.0 Phase 6b, ruling D8 a).

A rendered PDF is a binary: redaction never rewrites it, so a Drive link or home path baked
into it can only be removed by rendering it again. This renders each deck's *published* HTML
copy, after the payload family's own redaction rules (relink, ``inline_assets``), so the PDF
shows exactly what the HTML shows. The render happens once, into the reader-families staging
folder; isolated builds then copy the same bytes (a fresh render would differ by its
timestamps). A PDF that still holds a publishing marker is excluded and reported.

The workspace is never written: the HTML is rendered from a temporary folder, which is why it
must be self-contained (every image inline) before it is rendered.
"""

from __future__ import annotations

import fnmatch
import json
import re
import tempfile
from html.parser import HTMLParser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline._security.private_markers import has_publishing_marker_bytes
from conference_pipeline.publishing.bundle_redaction import RedactionError, apply_redaction_rules

MANIFEST_PATH = Path("storage") / "public_export_manifest.json"
PDF_NAME = "presentation.pdf"

# A renderer writes ``output_pdf`` and may return the loads it refused (the default renderer
# refuses every local one, 6b review r3 F6); any refused load refuses the PDF.
Renderer = Callable[..., Any]


class PdfRegenerationError(ValueError):
    def __init__(self, code: str, subject: str = "") -> None:
        super().__init__(f"{code}: {subject}" if subject else code)
        self.code = code
        self.subject = subject


@dataclass(frozen=True)
class RegeneratedPdfs:
    files: dict[str, bytes]  # "<deck>/presentation.pdf" -> bytes
    excluded: tuple[str, ...]


def _default_renderer(*, input_html: Path, output_pdf: Path) -> list[str]:
    from conference_pipeline.presentation.structured_video import export_pdf_with_playwright

    return export_pdf_with_playwright(input_html=input_html, output_pdf=output_pdf, refuse_local=True)


def _family(workspace: Path, family_id: str) -> Mapping[str, Any]:
    try:
        manifest = json.loads((workspace / MANIFEST_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PdfRegenerationError("pdf_manifest_unreadable", MANIFEST_PATH.as_posix()) from exc
    for family in manifest.get("families") or []:
        if isinstance(family, Mapping) and family.get("id") == family_id:
            return family
    raise PdfRegenerationError("pdf_family_unknown", family_id)


# What a page loads as it renders, read with an HTML parser (attributes quoted or not; a
# comment loads nothing): a source (img, script, video poster, object data, a lazy-loaded
# data-src), each srcset candidate, a <link>, an SVG <image>, <use> or <feImage> href, and in
# a style attribute or <style> each CSS url() and @import. An anchor's href loads nothing.
# Only an inline data: URI, an https URL or a fragment keeps the page self-contained;
# anything else, a file:// URL or an absolute path included, is a local dependency (review
# phase-6b-r1 F6, r2 F4 and F5). This read is the early refusal; the renderer is the
# enforcement, refusing any local load a script or a CSS escape makes (r3 F6).
_SOURCE_ATTRS = {"src", "data-src", "poster", "data"}
_HREF_LOADERS = {"link", "image", "use", "feimage"}
_CSS_URL = re.compile(r"""url\(\s*(["']?)(?P<ref>[^)"']*)\1\s*\)""", re.I)
_CSS_IMPORT = re.compile(r"""@import\s+(["'])(?P<ref>[^"']*)\1""", re.I)
_SRCSET_URL = re.compile(r"\s*(?P<url>[^\s,][^\s]*?)(?:,(?=\s|$)|\s+[^,]*(?:,|$)|$)")


def _srcset_urls(value: str) -> list[str]:
    """Each candidate URL of a srcset: a URL runs to whitespace, so a data URI's commas stay
    inside it (HTML's srcset grammar)."""
    return [m.group("url").rstrip(",") for m in _SRCSET_URL.finditer(value) if m.group("url")]


class _Loads(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refs: list[str] = []
        self._style = False

    def _css(self, text: str) -> None:
        self.refs += [m.group("ref") for p in (_CSS_URL, _CSS_IMPORT) for m in p.finditer(text)]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            value = value or ""
            if name in _SOURCE_ATTRS:
                self.refs.append(value)
            elif name == "srcset":
                self.refs += _srcset_urls(value)
            elif name in ("href", "xlink:href") and tag in _HREF_LOADERS:
                self.refs.append(value)
            elif name == "style":
                self._css(value)
            elif name == "srcdoc":  # a page of its own (6b review r3 F6)
                inner = _Loads()
                inner.feed(value)
                inner.close()
                self.refs += inner.refs
        self._style = tag == "style"

    handle_startendtag = handle_starttag

    def handle_endtag(self, tag: str) -> None:
        self._style = False

    def handle_data(self, data: str) -> None:
        if self._style:
            self._css(data)


# Chromium stamps the render time in the PDF's CreationDate and ModDate, so each regeneration
# gave new bytes (F-6b-27). The 14 date digits are pinned in place: same length, so the
# cross-reference offsets stay valid, and the rest of the file is untouched.
PINNED_PDF_DATE = b"20000101000000"
_PDF_DATE = re.compile(rb"(/(?:CreationDate|ModDate)\s*\(D(?::|\\072))\d{14}")  # ":" plain or octal-escaped


def _pin_dates(data: bytes) -> bytes:
    return _PDF_DATE.sub(lambda m: m.group(1) + PINNED_PDF_DATE, data)


def _self_contained(ref: str) -> bool:
    ref = ref.strip()
    return not ref or ref.lower().startswith(("data:", "https:", "#"))


def _local_resources(text: str) -> list[str]:
    parser = _Loads()
    parser.feed(text)
    parser.close()
    return [ref for ref in parser.refs if not _self_contained(ref)]


def regenerate_deck_pdfs(workspace: Path, spec: Mapping[str, Any], *, render: Renderer | None = None) -> RegeneratedPdfs:
    """Render every deck HTML matched by ``spec["html"]`` in ``spec["family"]``'s source."""
    workspace = Path(workspace)
    family = _family(workspace, str(spec.get("family") or ""))
    source = workspace / str(family.get("private_path") or "")
    pattern = str(spec.get("html") or "*/presentation.html")
    rules = family.get("redaction_rules") or []
    render = render or _default_renderer
    files: dict[str, bytes] = {}
    excluded: list[str] = []
    for html in sorted(p for p in source.rglob("*.html") if fnmatch.fnmatch(p.relative_to(source).as_posix(), pattern)):
        rel = html.relative_to(source).as_posix()
        original = html.read_bytes()
        try:
            outcome = apply_redaction_rules(original, rel, rules, workspace=workspace, source_path=html)
        except RedactionError as exc:
            raise PdfRegenerationError("pdf_html_redaction_failed", f"{rel}: {exc}") from exc
        published = outcome.content if outcome.content is not None else original
        text = published.decode("utf-8", errors="replace")
        left = _local_resources(text)
        if left:
            raise PdfRegenerationError("pdf_html_not_self_contained", f"{rel} loads {left[0]}")
        pdf_rel = (Path(rel).parent / PDF_NAME).as_posix()
        with tempfile.TemporaryDirectory(prefix="release-pdf-") as tmp:
            page = Path(tmp) / "page" / "presentation.html"
            out = Path(tmp) / "out" / PDF_NAME
            page.parent.mkdir()
            out.parent.mkdir()
            page.write_bytes(published)
            refused = list(render(input_html=page, output_pdf=out) or [])
            if refused:
                raise PdfRegenerationError("pdf_html_not_self_contained", f"{rel} loads {refused[0]}")
            data = _pin_dates(out.read_bytes()) if out.is_file() else b""
        if not data or has_publishing_marker_bytes(data):
            excluded.append(pdf_rel)
            continue
        files[pdf_rel] = data
    return RegeneratedPdfs(files=files, excluded=tuple(excluded))


__all__ = ["PdfRegenerationError", "RegeneratedPdfs", "regenerate_deck_pdfs"]
