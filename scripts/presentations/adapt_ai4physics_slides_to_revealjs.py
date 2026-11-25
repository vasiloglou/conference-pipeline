#!/usr/bin/env python3
"""Adapt the AI4Physics hand-authored slide deck to canonical Reveal.js markdown.

`cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md` is a delivery-final,
hand-authored deck that uses its own format:

    <!-- slide_id: some-id -->
    # Visible H1 title

    Optional visible body (subtitle, bullets, tables, `![alt](path)` figures).

    <!--
    slide_kind: references
    speaker_notes_source: narrative.md#slide-N-some-id
    speaker_notes: |
      Multi-line speaker notes...
    -->

    ---

`scripts/presentations/md_to_revealjs.py` (the shared, branded Reveal.js
renderer used for every other conference deck) does not understand this
format -- it expects canonical blocks shaped like:

    ## Slide N: Title

    <body>

    <!--
    SLIDE_TYPE: content
    SPEAKER_NOTES: ...
    -->

    ---

This module bridges the two **without forking or modifying the renderer**
(a locked decision -- see `.planning/phases/40-reveal-js-rendered-delivery-artifacts/
40-CONTEXT.md`): it parses the authored format, maps each slide's `slide_kind`
to the renderer's `SLIDE_TYPE`, and re-emits canonical markdown that
`md_to_revealjs.parse_slides()` consumes unchanged.

Deliberately NOT using a YAML parser for the authored metadata comment: a real
YAML parser (`yaml.safe_load`) raises `yaml.scanner.ScannerError` on 2 of the
49 real blocks in `slides.md`, because some free-text values contain an
unescaped ": " (e.g. "The ratio is roughly 3: to 1..."). Instead this module
reimplements the same line/continuation scanning style as the renderer's own
`_parse_metadata()` (`md_to_revealjs.py` ~lines 348-363), generalized to the
authored deck's lowercase `snake_case:` keys.

Figure references (`![alt](path)`) and speaker-notes text are passed through
UNCHANGED -- the renderer resolves figure paths (`Path.resolve()` + existence
check) and HTML-escapes notes text itself; pre-resolving or pre-escaping here
would double-handle both.

CLI usage:

    python scripts/presentations/adapt_ai4physics_slides_to_revealjs.py \\
        --input cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md \\
        --canonical-out cross-conference-talks/ai4physics2026/decks/ai4physics-talk/_canonical_slides.md

IMPORTANT: `--canonical-out` MUST be written at the same directory depth as
`--input` (i.e. colocated with `slides.md`, not in /tmp or a nested build
directory). The authored deck's figure references are relative paths like
`../../assets/some-figure.svg`; `md_to_revealjs.py` resolves those paths
relative to the *canonical file's own directory*, not the original
`slides.md`'s directory. Writing the canonical file anywhere else silently
breaks every figure reference.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# slide_kind -> SLIDE_TYPE mapping (locked mapping, see 40-CONTEXT.md)
# ---------------------------------------------------------------------------

_SECTION_SLIDE_KINDS = {"section", "native-divider"}


def map_slide_kind_to_slide_type(slide_kind: str) -> str:
    """Map an authored `slide_kind` value to the renderer's `SLIDE_TYPE`.

    `title` -> `title`; `section` and `native-divider` -> `section`;
    everything else (`references`, `trend-findings`, `cost-breakdown`, etc.)
    -> `content`.
    """
    if slide_kind == "title":
        return "title"
    if slide_kind in _SECTION_SLIDE_KINDS:
        return "section"
    return "content"


# ---------------------------------------------------------------------------
# Authored metadata comment parser (NOT a YAML parser -- see module docstring)
# ---------------------------------------------------------------------------

_KEY_LINE_RE = re.compile(r"^([a-z][a-z0-9_]*):\s*(.*)$")


def _parse_authored_metadata(comment_text: str) -> dict[str, str]:
    """Line/continuation scanner for the authored deck's lowercase metadata.

    Modeled on `md_to_revealjs.py`'s `_parse_metadata()`: a line matching
    `^key:\\s*value` at column 0 (after per-line stripping) starts a new
    field; any following line that does not itself start a new top-level key
    is a continuation of the current field's value (this also correctly
    handles YAML-style `key: |` block-scalar bodies, and plain indented
    `- list` items under keys this module doesn't care about, e.g.
    `evidence:`).

    A single-line value containing an internal `": "` (e.g.
    `speaker_notes: The ratio is roughly 3: to 1...`) round-trips correctly
    because the key regex only ever matches at the start of a physical line;
    a colon appearing mid-line is just ordinary text.
    """
    out: dict[str, str] = {}
    current_key: str | None = None
    current_val: list[str] = []

    def flush() -> None:
        if current_key is not None:
            out[current_key] = " ".join(v for v in current_val if v.strip() != "").strip()

    for raw_line in comment_text.strip("\n").split("\n"):
        line = raw_line.strip()
        match = _KEY_LINE_RE.match(line)
        if match:
            flush()
            current_key = match.group(1)
            first_val = match.group(2).strip()
            # `key: |` is a block-scalar marker with no inline value -- the
            # real value lives entirely in the continuation lines that follow.
            current_val = [] if first_val == "|" else ([first_val] if first_val else [])
        elif current_key is not None:
            current_val.append(line)
    flush()
    return out


# ---------------------------------------------------------------------------
# Authored-format parser
# ---------------------------------------------------------------------------

_SLIDE_ID_RE = re.compile(r"^<!--\s*slide_id:\s*(.+?)\s*-->")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_METADATA_MARKER_RE = re.compile(r"^\s*slide_kind:\s*", re.MULTILINE)


def parse_authored_slides(md_text: str) -> list[dict]:
    """Parse the authored deck's markdown into a list of slide dicts.

    Each dict has keys: `slide_id`, `title` (the visible `# H1` text),
    `body` (visible content after the H1, excluding the trailing metadata
    comment), `slide_kind`, `speaker_notes`.

    Blocks are split on a literal `\\n---\\n`, mirroring exactly the
    separator `md_to_revealjs.parse_slides()` already requires.
    """
    slides: list[dict] = []
    raw_blocks = re.split(r"\n---\n", md_text)

    for raw_block in raw_blocks:
        block = raw_block.strip()
        if not block:
            continue

        slide_id_match = _SLIDE_ID_RE.match(block)
        slide_id = slide_id_match.group(1).strip() if slide_id_match else ""

        h1_match = _H1_RE.search(block)
        if not h1_match:
            # No visible H1 in this block -- not a real authored slide block
            # (e.g. stray content). Skip rather than fabricate a title.
            continue
        title = h1_match.group(1).strip()

        # Identify the trailing metadata comment: the one HTML comment in
        # this block whose body contains a `slide_kind:` line. This is
        # distinct from the leading single-line `<!-- slide_id: ... -->`
        # comment, which never contains `slide_kind:`.
        metadata: dict[str, str] = {}
        metadata_comment_match = None
        for comment_match in _COMMENT_RE.finditer(block):
            comment_body = comment_match.group(1)
            if _METADATA_MARKER_RE.search(comment_body):
                metadata = _parse_authored_metadata(comment_body)
                metadata_comment_match = comment_match

        content_start = h1_match.end()
        if metadata_comment_match:
            body = block[content_start:metadata_comment_match.start()].strip()
        else:
            body = block[content_start:].strip()

        slides.append({
            "slide_id": slide_id,
            "title": title,
            "body": body,
            "slide_kind": metadata.get("slide_kind", ""),
            "speaker_notes": metadata.get("speaker_notes", ""),
        })

    return slides


# ---------------------------------------------------------------------------
# Canonical markdown emitter
# ---------------------------------------------------------------------------

def emit_canonical_markdown(slides: list[dict]) -> str:
    """Emit canonical `## Slide N: Title` markdown for the shared renderer.

    Numbers slides sequentially 1..len(slides) in authored order. Passes
    figure references and body content through UNCHANGED. Discards the
    authored deck's own lowercase-key metadata comment entirely and
    re-serializes a synthetic ALL-CAPS `SLIDE_TYPE`/`SPEAKER_NOTES` comment.
    """
    blocks: list[str] = []

    for index, slide in enumerate(slides, start=1):
        title = slide["title"]
        body = (slide.get("body") or "").strip()
        slide_type = map_slide_kind_to_slide_type(slide.get("slide_kind", ""))

        if slide_type == "title":
            # Unlike section/content slides, the renderer's `_render_title`
            # has no fallback that injects a heading -- it renders the body
            # verbatim. So the visible H1 must be part of the body itself,
            # or the title slide would show no title text at all.
            body = f"# {title}\n\n{body}".strip() if body else f"# {title}"

        block_lines = [f"## Slide {index}: {title}", ""]
        if body:
            block_lines.append(body)
            block_lines.append("")

        comment_lines = [f"SLIDE_TYPE: {slide_type}"]
        notes = (slide.get("speaker_notes") or "").strip()
        if notes:
            comment_lines.append(f"SPEAKER_NOTES: {notes}")
        block_lines.append("<!--\n" + "\n".join(comment_lines) + "\n-->")

        blocks.append("\n".join(block_lines).strip())

    return "\n\n---\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the AI4Physics hand-authored slides.md into canonical "
            "markdown for scripts/presentations/md_to_revealjs.py. "
            "--canonical-out MUST be colocated (same directory depth) with "
            "--input so relative figure paths (../../assets/...) resolve "
            "unchanged."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the authored slides.md.")
    parser.add_argument(
        "--canonical-out",
        type=Path,
        required=True,
        help=(
            "Output path for the canonical intermediate markdown. Recommended: "
            "colocated with --input, e.g. "
            "cross-conference-talks/ai4physics2026/decks/ai4physics-talk/_canonical_slides.md"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1

    md_text = input_path.read_text(encoding="utf-8")
    slides = parse_authored_slides(md_text)
    canonical_text = emit_canonical_markdown(slides)

    canonical_out = args.canonical_out.expanduser().resolve()
    canonical_out.parent.mkdir(parents=True, exist_ok=True)
    canonical_out.write_text(canonical_text, encoding="utf-8")

    print(f"Wrote {len(slides)} slides to {canonical_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
