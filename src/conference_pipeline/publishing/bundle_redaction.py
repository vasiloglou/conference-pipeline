"""Redact a file's bundle copy by a family's ``redaction_rules`` (v6.0 Phase 6, spec section 6.5).

The scrub policy (``docs/operations/public-release-scrub-policy.md``) says private values are
removed from the bundle copy, never from the workspace original (R9). The exporter calls
``apply_redaction_rules`` while planning, so the planned digest, the written bytes and
``CHECKSUMS.json`` all describe the redacted copy.

Rule objects (strings are legacy names and do nothing):

- ``json_field``: replace the value at ``pointer`` with ``replacement``;
- ``absolute_path``: paths inside the workspace become workspace-relative; other home paths
  become ``<redacted:path>``;
- ``relink``: a Drive link in an HTML anchor is rewritten through ``link_map`` (a workspace
  JSON file of Drive URL -> https URL); an unmapped one is dropped and its text kept. A
  relative ``href`` the map names exactly is rewritten too; an unmapped one is left alone;
- ``marker_text``: publishing markers become ``<redacted:RULE>``;
- ``inline_assets``: each local image an HTML file loads (``src``/``href`` with an image suffix)
  becomes a ``data:`` URI, so the published deck is one self-contained file (owner ruling D21,
  Phase 6b). The image must lie inside the project, reached through no link, and hold no
  publishing marker (base64 would hide it from every later scan). It always runs last, so no
  other rule rewrites base64 text.

- ``pdf_links``: in a PDF, each link annotation that is not a public https link is removed;
  the pages are unchanged (6b, ICLR 2025);
- ``drop_preamble``: text above a Markdown file's first ``# `` title is dropped (6b, ICLR
  2025: a model's chat line above a presenter script).

No other binary file is rewritten. If one holds a publishing marker it is reported, for
regeneration or exclusion (D8).
"""

from __future__ import annotations

import base64
import fnmatch
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline._security.private_markers import has_publishing_marker_bytes, redact_publishing_markers

# An absolute path that is not part of a URL: at least two segments, not preceded by a scheme,
# host or path character (review phase-6a-r1 F5: /srv, /Volumes, /private ... were missed).
# A segment may hold single inner spaces ("/Volumes/Private Drive/x"). A colon may precede the
# path ("path:/srv/x"); a URL's own slashes never start one (review phase-6a-r2 F4).
ABSOLUTE_PATH_START = re.compile(r"""(?<![\w/.\-~])/(?=[^\s/])""")
PATH_DELIMITERS = frozenset('"\'`<>()[]{}|\n')
# Every inline form: a bare or <angle> destination, and a "double", 'single' or (paren) title
# (review phase-6a-r3 F5).
MARKDOWN_DRIVE_LINK = re.compile(
    r"""\[([^\]]*)\]\(\s*<?(https?://drive\.google\.com/[^)\s>]*)>?(\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?\s*\)"""
)
BARE_DRIVE_URL = re.compile(r"""(?P<space>[ \t]?)(?P<open>\()?(?P<url>https?://drive\.google\.com/[^\s)<>"']*)(?P<close>\))?""")
MARKDOWN_DRIVE_REFERENCE = re.compile(
    r"""^[ \t]{0,3}\[([^\]]+)\]:[ \t]*<?(https?://drive\.google\.com/[^\s>]*)>?([ \t]+"[^"]*")?[ \t]*$""",
    re.MULTILINE,
)
DRIVE_ANCHOR = re.compile(
    r"""<a\b[^>]*\bhref=(["'])(https?://drive\.google\.com/[^"']*)\1[^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)
# A relative link in an HTML attribute (``href="../01_x/presentation.html"``), rewritten only
# when the link map names it exactly (6b: ICLR 2026's executive decks link topic decks this way,
# and a PDF rendered from a temporary folder bakes such links in as file:// paths).
RELATIVE_HREF = re.compile(r"""(\bhref=)(["'])(\.\.?/[^"']*)\2""", re.IGNORECASE)
KINDS = ("json_field", "absolute_path", "relink", "marker_text", "inline_assets", "figure_links", "pdf_links",
         "drop_preamble")
# The first Markdown title line (``# Title``).
FIRST_HEADING = re.compile(r"^# \S", re.MULTILINE)
# A local image in Markdown: ``![alt](path)`` whose path has no scheme.
MARKDOWN_LOCAL_IMAGE = re.compile(r"!\[[^\]]*\]\((?![a-zA-Z][\w+.\-]*:)(?P<ref>[^)\s]+)\)")
IMAGE_TYPES = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".webp": "image/webp"}
# A local reference: no scheme, not absolute, not a fragment, not already inline.
LOCAL_ASSET = re.compile(r"""(?P<attr>\b(?:src|href))=(?P<q>["'])(?P<ref>(?![a-zA-Z][\w+.\-]*:|/|#)[^"'?#]+)(?P=q)""")
# Text by type: only these suffixes are ever rewritten. Anything else is binary, never
# rewritten, only checked for markers (review phase-6a-r1 F4: an unlisted binary that happened
# to decode as UTF-8 was rewritten and its size broke).
TEXT_SUFFIXES = frozenset({
    ".md", ".markdown", ".txt", ".html", ".htm", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
    ".xml", ".svg", ".css", ".js", ".tex", ".bib", ".srt", ".vtt",
})


class RedactionError(ValueError):
    """A rule is malformed or its link map cannot be read."""


@dataclass(frozen=True)
class RedactionOutcome:
    content: bytes | None
    kinds: tuple[str, ...]
    replacements: int
    binary_marker: bool


def _matches(rel: str, glob: str) -> bool:
    return fnmatch.fnmatch(rel, glob) or (glob.startswith("**/") and fnmatch.fnmatch(rel, glob[3:]))


def _pointer_set(doc: Any, pointer: str, value: str) -> int:
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.lstrip("/").split("/")]
    node = doc
    for part in parts[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return 0
    last = parts[-1]
    if isinstance(node, dict) and last in node and node[last] != value:
        node[last] = value
        return 1
    if isinstance(node, list) and last.isdigit() and int(last) < len(node) and node[int(last)] != value:
        node[int(last)] = value
        return 1
    return 0


def _absolute_paths(text: str, workspace: Path) -> tuple[str, int]:
    count = 0
    for root in sorted({str(workspace), str(workspace.resolve())}, key=len, reverse=True):
        prefix = root.rstrip("/") + "/"
        count += text.count(prefix)
        text = text.replace(prefix, "")
    text, n = _redact_absolute_paths(text)
    return text, count + n


def _path_end(text: str, start: int) -> int:
    """Where a path that starts at ``start`` ends (review phase-6a-r3 F4).

    A path runs over every character but a delimiter. At whitespace it continues only when
    the next word still looks like part of a path (holds a "/" or a "."), so
    "/srv/a/customer secret.json" is one path and "/srv/a/x.json done" stops before "done".
    """
    i = start
    while i < len(text):
        ch = text[i]
        if ch in PATH_DELIMITERS:
            return i
        if ch.isspace():
            j = i
            while j < len(text) and text[j] in " \t":
                j += 1
            k = j
            while k < len(text) and not text[k].isspace() and text[k] not in PATH_DELIMITERS:
                k += 1
            word = text[j:k]
            if j == i or not word or not ("/" in word or "." in word) or "://" in word:
                return i
            i = k
            continue
        i += 1
    return i


def _redact_absolute_paths(text: str) -> tuple[str, int]:
    out: list[str] = []
    pos = 0
    count = 0
    for match in ABSOLUTE_PATH_START.finditer(text):
        start = match.start()
        if start < pos:
            continue
        end = _path_end(text, start)
        candidate = text[start:end].rstrip(".,;:")
        if candidate.count("/") < 2:
            continue
        out.append(text[pos:start])
        out.append("<redacted:path>")
        pos = start + len(candidate)
        count += 1
    out.append(text[pos:])
    return "".join(out), count


def _relink(text: str, link_map: Mapping[str, str]) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        target = link_map.get(match.group(2))
        if isinstance(target, str) and target.startswith("https://"):
            return match.group(0).replace(match.group(2), target)
        return match.group(3)

    def replace_markdown(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        target = link_map.get(match.group(2))
        if isinstance(target, str) and target.startswith("https://"):
            return f"[{match.group(1)}]({target}{match.group(3) or ''})"
        return match.group(1)

    dropped_labels: set[str] = set()

    def replace_reference(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        target = link_map.get(match.group(2))
        if isinstance(target, str) and target.startswith("https://"):
            return f"[{match.group(1)}]: {target}{match.group(3) or ''}"
        dropped_labels.add(match.group(1).lower())
        return ""

    def replace_bare(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        target = link_map.get(match.group("url"))
        if isinstance(target, str) and target.startswith("https://"):
            return f"{match.group('space')}{match.group('open') or ''}{target}{match.group('close') or ''}"
        return ""  # an unmapped bare link goes, with its parentheses and leading space

    def replace_relative(match: re.Match[str]) -> str:
        nonlocal count
        target = link_map.get(match.group(3))
        if isinstance(target, str) and target.startswith("https://"):
            count += 1
            return f"{match.group(1)}{match.group(2)}{target}{match.group(2)}"
        return match.group(0)

    text = DRIVE_ANCHOR.sub(replace, text)
    text = RELATIVE_HREF.sub(replace_relative, text)
    text = MARKDOWN_DRIVE_LINK.sub(replace_markdown, text)
    text = MARKDOWN_DRIVE_REFERENCE.sub(replace_reference, text)
    # Bare Drive URLs left after the link forms, optionally in parentheses (6b: ICML's
    # executive presenter scripts cite "Source: X (https://drive.google.com/...)").
    text = BARE_DRIVE_URL.sub(replace_bare, text)
    if dropped_labels:
        # A reference to a dropped definition keeps only its text (review phase-6a-r2 F5).
        text = re.sub(r"\[([^\]]*)\]\[([^\]]*)\]",
                      lambda m: m.group(1) if (m.group(2) or m.group(1)).lower() in dropped_labels else m.group(0), text)
    return text, count


def _title_key(title: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", title.casefold()).split())


def _figure_links(text: str, source_path: Path, workspace: Path, catalog_rel: str) -> tuple[str, int]:
    """Each local figure in deck Markdown becomes a line naming the paper it was cropped from
    and linking that paper's official copy (owner ruling D25, NeurIPS 2025: paper figures are
    the papers' content, linked, never copied). The paper comes from the deck's
    ``asset_inventory.json`` (``source_slide`` + ``image_filename`` -> ``source_paper``), the
    link from the release catalog by title; without either, a plain notice stays."""
    figures: dict[str, str] = {}
    inventory = source_path.parent / "asset_inventory.json"
    if inventory.is_file():
        try:
            data = json.loads(inventory.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RedactionError(f"figure_links: unreadable {inventory.name}") from exc
        for figure in data.get("figures") or [] if isinstance(data, Mapping) else []:
            if isinstance(figure, Mapping) and figure.get("source_paper"):
                key = f"assets/figures/{figure.get('source_slide')}/{figure.get('image_filename')}"
                figures[key] = " ".join(str(figure["source_paper"]).split())
    links: dict[str, str] = {}
    catalog = workspace / catalog_rel
    if catalog_rel and catalog.is_file():
        for line in catalog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                url = row.get("official_url")
                if isinstance(url, str) and url.startswith("https://"):
                    links.setdefault(_title_key(str(row.get("title") or "")), url)
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        ref = match.group("ref").lstrip("./")
        if ref.startswith("assets/generated/"):
            # A figure the pipeline drew for the deck, not a paper's (6b, ICLR 2025).
            return "*A figure made for this deck, not included in this release.*"
        title = figures.get(ref)
        if title is None:
            return "*A figure from a cited paper, not reproduced in this release.*"
        url = links.get(_title_key(title))
        if url:
            return f"*Figure from [{title}]({url}), not reproduced in this release.*"
        return f'*Figure from "{title}", not reproduced in this release.*'

    return MARKDOWN_LOCAL_IMAGE.sub(replace, text), count


def project_root_of(workspace: Path) -> Path:
    """The checkout that holds ``conferences/<slug>``; the workspace itself otherwise."""
    workspace = Path(os.path.abspath(workspace))
    return workspace.parents[1] if workspace.parent.name == "conferences" else workspace


def _asset_bytes(ref: str, source_path: Path, root: Path) -> bytes:
    """An image's bytes, refused unless it lies inside ``root`` through real folders only."""
    lexical = Path(os.path.normpath(source_path.parent / ref))
    if not lexical.is_relative_to(root):
        raise RedactionError(f"inline_assets: {ref} lies outside the project")
    current = root
    for part in lexical.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise RedactionError(f"inline_assets: {ref} is reached through a link")
    if not lexical.is_file():
        raise RedactionError(f"inline_assets: {ref} does not exist")
    data = lexical.read_bytes()
    if has_publishing_marker_bytes(data):
        raise RedactionError(f"inline_assets: {ref} holds a publishing marker")
    return data


def _inline_assets(text: str, source_path: Path, root: Path) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        mime = IMAGE_TYPES.get(Path(match.group("ref")).suffix.lower())
        if mime is None:
            return match.group(0)
        count += 1
        encoded = base64.b64encode(_asset_bytes(match.group("ref"), source_path, root)).decode("ascii")
        return f'{match.group("attr")}={match.group("q")}data:{mime};base64,{encoded}{match.group("q")}'

    return LOCAL_ASSET.sub(replace, text), count


def _pdf_links(data: bytes) -> tuple[bytes | None, int]:
    """A PDF's link annotations that are not public https links (a ``file://`` home path baked
    in by the renderer, a private link) removed; the pages are unchanged (6b, ICLR 2025). The
    copy is written from the original's objects, so the same input gives the same bytes."""
    import io
    import logging

    import pypdf
    from pypdf.generic import ArrayObject, NameObject

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        writer = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(data)))
    except Exception as exc:  # pypdf raises several error types for a damaged file
        raise RedactionError(f"pdf_links: unreadable PDF ({type(exc).__name__})") from exc
    removed = 0
    for page in writer.pages:
        annots = page.get("/Annots")
        if not annots:
            continue
        keep = []
        for annot in annots:
            action = annot.get_object().get("/A")
            uri = action.get("/URI") if action else None
            if uri is not None and (not str(uri).startswith("https://") or has_publishing_marker_bytes(str(uri).encode())):
                removed += 1
                continue
            keep.append(annot)
        if len(keep) != len(annots):
            page[NameObject("/Annots")] = ArrayObject(keep)
    if not removed:
        return None, 0
    # A removed annotation is unreferenced, not gone: written out, its home path stayed in
    # the file (found by 6b review r3's whole-object PDF read). Orphans are dropped.
    # pypdf 7 removed the old keyword names; older releases have only them (v6.0 Phase 7,
    # found on a fresh install).
    try:
        writer.compress_identical_objects(remove_duplicates=False, remove_unreferenced=True)
    except TypeError:
        writer.compress_identical_objects(remove_identicals=False, remove_orphans=True)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue(), removed


def apply_redaction_rules(data: bytes, rel: str, rules: Sequence[Any], *, workspace: Path,
                          source_path: Path | None = None) -> RedactionOutcome:
    """The redacted bytes of one file (or ``None`` when nothing changed).

    ``source_path`` is the workspace original, needed by ``inline_assets`` to resolve the
    images the file loads."""
    objects = [rule for rule in rules if isinstance(rule, Mapping)]
    if not objects:
        return RedactionOutcome(None, (), 0, False)
    for rule in objects:
        if rule.get("kind") not in KINDS or not rule.get("glob"):
            raise RedactionError(f"malformed redaction rule: {dict(rule)}")
    if Path(rel).suffix.lower() == ".pdf" and any(r["kind"] == "pdf_links" and _matches(rel, str(r["glob"])) for r in objects):
        new, n = _pdf_links(data)
        return RedactionOutcome(new, ("pdf_links",) if n else (), n, has_publishing_marker_bytes(new or data))
    if Path(rel).suffix.lower() not in TEXT_SUFFIXES:
        return RedactionOutcome(None, (), 0, has_publishing_marker_bytes(data))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return RedactionOutcome(None, (), 0, has_publishing_marker_bytes(data))
    kinds: list[str] = []
    total = 0
    # inline_assets last, whatever the listed order: no rule may rewrite base64 text.
    for rule in sorted(objects, key=lambda r: r["kind"] == "inline_assets"):
        if not _matches(rel, str(rule["glob"])):
            continue
        kind = rule["kind"]
        n = 0
        if kind == "json_field":
            if not rule.get("pointer") or not rule.get("replacement"):
                raise RedactionError("json_field needs pointer and replacement")
            try:
                doc = json.loads(text)
            except json.JSONDecodeError:
                continue
            n = _pointer_set(doc, str(rule["pointer"]), str(rule["replacement"]))
            if n:
                text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        elif kind == "absolute_path":
            text, n = _absolute_paths(text, workspace)
        elif kind == "relink":
            map_path = workspace / str(rule.get("link_map") or "")
            try:
                link_map = json.loads(map_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RedactionError(f"relink link_map unreadable: {rule.get('link_map')}") from exc
            text, n = _relink(text, link_map)
        elif kind == "marker_text":
            redacted = redact_publishing_markers(text)
            n = int(redacted != text)
            text = redacted
        elif kind == "drop_preamble":
            heading = FIRST_HEADING.search(text)
            n = int(bool(heading and heading.start() > 0))
            if n:
                text = text[heading.start():]
        elif kind == "figure_links":
            if source_path is None:
                raise RedactionError("figure_links needs the source file's path")
            text, n = _figure_links(text, Path(os.path.abspath(source_path)), Path(workspace), str(rule.get("catalog") or ""))
        elif kind == "inline_assets":
            if source_path is None:
                raise RedactionError("inline_assets needs the source file's path")
            text, n = _inline_assets(text, Path(os.path.abspath(source_path)), project_root_of(workspace))
        if n:
            kinds.append(kind)
            total += n
    new = text.encode("utf-8")
    return RedactionOutcome(new if new != data else None, tuple(kinds), total, False)
