#!/usr/bin/env python3
"""Derive a scannable presenter cheat sheet from narrative.md.

narrative.md is the source-of-truth presenter companion: full Establishes /
Script / Provenance / Figure-enrichment blocks per slide, written for archival
completeness. That density is exactly what makes it hard to track your place
in while actually speaking. This script derives a leaner artifact for glancing
at *while* presenting:

  - Drops Provenance and Figure/enrichment leads (not needed live).
  - Keeps Establishes (as a one-line reminder), Delivery note / Connect-back /
    Only-quantitative-claims callouts (they are presenting instructions).
  - Reflows each Script paragraph into one bullet per sentence, so the eye can
    track progress line-by-line instead of losing its place in a paragraph.
    This is a pure re-wrapping transform: no words are added, removed, or
    paraphrased -- only where line breaks fall changes. Sentence boundaries
    are detected by punctuation ([.!?]) followed by whitespace and a capital
    letter or opening quote; this never fires on mid-number decimals (e.g.
    "$492.49") because there is no whitespace between the digits and the
    period there, so a worst-case misfire only produces an odd line break,
    never altered content.
  - Flags the 15 references-kind "backup" slides (narration_status: backup in
    slides.md) so the presenter knows to flip past them in the live flow.

Usage:
    .venv/bin/python scripts/presentations/build_ai4physics_presenter_cheatsheet.py \\
        --narrative cross-conference-talks/ai4physics2026/decks/ai4physics-talk/narrative.md \\
        --slides-md cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md \\
        --output cross-conference-talks/ai4physics2026/decks/ai4physics-talk/presenter_cheatsheet.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_SLIDE_BLOCK_RE = re.compile(
    r"^## Slide (?P<num>\d+[a-z]?) — (?P<title>.+?) \(`(?P<slide_id>[a-z0-9-]+)`\)"
    r" · (?P<tag>.+)$",
    re.MULTILINE,
)
# Any bold header line, e.g. "**Script:**" or the parenthetical-qualified
# variants that appear on a few slides ("**Script (spoken with verified fills;
# keep the honesty flags):**", "**Provenance (verified numbers; several
# manifests live in the populated main\ncheckout, ...):**" -- note this one
# soft-wraps across two physical lines, matching the file's ~80-column prose
# wrapping convention). Non-greedy `.+?` with DOTALL is what makes this robust
# to both a parenthetical qualifier AND a mid-header line wrap -- it matches
# up to the FIRST ":**" after the opening "**", however many lines that spans.
# A stricter single-line/charset pattern silently fails to recognize the
# header as a field boundary, letting that field's body swallow every
# subsequent field (including Script) into one unparsed blob.
_FIELD_HEADER_RE = re.compile(r"^\*\*(?P<field>.+?):\*\*[ \t]*\n?", re.MULTILINE | re.DOTALL)


def _extract_fields(block_text: str) -> dict[str, str]:
    """Map each slide block's bold field name -> its body text.

    Field names are normalized by stripping any parenthetical qualifier
    ("Script (spoken with verified fills...)" -> "Script") so lookups by
    canonical name ("Script", "Establishes", ...) work regardless of which
    variant a given slide used. A field's body runs to the next header, the
    trailing HTML-comment divider, or end of block -- whichever is first.
    """
    headers = list(_FIELD_HEADER_RE.finditer(block_text))
    fields: dict[str, str] = {}
    for i, h in enumerate(headers):
        body_start = h.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(block_text)
        body = block_text[body_start:body_end]
        body = re.split(r"\n<!-- =+", body)[0]  # drop the trailing slide divider
        # DOTALL (not just `$`): the field text can itself wrap across lines
        # (e.g. the Provenance header above), so ".*" must cross newlines to
        # strip the whole parenthetical qualifier, not just its first line.
        name = re.sub(r"\s*\(.*", "", h.group("field"), flags=re.DOTALL).strip()
        name = re.sub(r"\s+", " ", name)  # collapse any internal wrap-newline
        if name not in fields:  # first occurrence wins
            fields[name] = body
    return fields
# Sentence boundary: punctuation, optionally followed by a closing quote,
# then whitespace, then an uppercase letter or opening quote. Never matches
# inside a bare decimal ("$492.49") because there is no whitespace directly
# after that period. Both lookbehind alternatives are zero-width, so a
# trailing quote mark (e.g. '...regime." When ...') is never consumed by the
# split -- it stays attached to the sentence that owns it, instead of being
# silently discarded (re.split() drops any characters inside the matched span).
_SENTENCE_SPLIT_RE = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]"))\s+(?=[A-Z"])')


def _backup_slide_ids(slides_md_text: str) -> set[str]:
    """Return slide_ids marked `narration_status: backup` in slides.md."""
    ids = set()
    for block in re.split(r"\n---\n", slides_md_text):
        if "narration_status: backup" not in block:
            continue
        m = re.search(r"slide_id:\s*([a-z0-9-]+)", block)
        if m:
            ids.add(m.group(1))
    return ids


def _unwrap_blockquote(raw_field_body: str) -> str:
    """Strip leading '> ' from each blockquote line; keep blank-line paragraph
    breaks (a bare '>' line) as a paragraph separator marker '\n\n'."""
    lines = raw_field_body.strip("\n").split("\n")
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == ">":
            out_lines.append("")  # paragraph break
        elif stripped.startswith(">"):
            out_lines.append(stripped[1:].strip())
        else:
            out_lines.append(stripped)
    # Join wrapped lines within a paragraph, keep blank lines as breaks.
    paragraphs, current = [], []
    for line in out_lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _script_to_bullets(script_text: str) -> list[str]:
    """Split each paragraph into one bullet per sentence."""
    bullets = []
    for paragraph in script_text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                bullets.append(sentence)
    return bullets


def _reflow_plain(text: str) -> str:
    """Reflow a non-blockquote field (Establishes, Delivery note, ...) to one line."""
    return " ".join(line.strip() for line in text.strip().split("\n") if line.strip())


def _tag_suffix(tag: str) -> str:
    """Compact a header tag into a short bracketed flag, or '' for plain timing."""
    if re.fullmatch(r"\d+:\d\d", tag):
        return ""
    label = tag.split(" ", 1)[0].split("(", 1)[0].strip().rstrip(" -")
    return f"  ⚑ {label}"


def build_cheatsheet(narrative_text: str, slides_md_text: str) -> str:
    backup_ids = _backup_slide_ids(slides_md_text)

    matches = list(_SLIDE_BLOCK_RE.finditer(narrative_text))
    out = [
        "# AI4Physics Talk — Presenter Cheat Sheet",
        "",
        "Derived from `narrative.md` for glancing at while presenting: one bullet",
        "per sentence (nothing paraphrased), provenance/figure-lead blocks dropped.",
        "`⏭ BACKUP` slides are references slides — flip past them in the live flow",
        "unless the audience asks. On-screen Reveal slide numbers now match these",
        "deck positions (1..49) exactly, so no mental remapping is needed while",
        "speaking. Regenerate with",
        "`scripts/presentations/build_ai4physics_presenter_cheatsheet.py` after any",
        "narrative.md change.",
        "",
        "---",
        "",
    ]

    for i, m in enumerate(matches):
        num, title, slide_id, tag = (
            m.group("num"),
            m.group("title"),
            m.group("slide_id"),
            m.group("tag"),
        )
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(narrative_text)
        block_text = narrative_text[m.end():block_end]

        fields = _extract_fields(block_text)

        is_backup = slide_id in backup_ids
        header_flag = " ⏭ BACKUP (not narrated)" if is_backup else _tag_suffix(tag)
        # Plain "m:ss" tags get a clock emoji; special tags (SPECULATIVE,
        # CLOSER, MIXED, OVERFLOW, ...) are printed verbatim instead -- showing
        # "⏱ SPECULATIVE" would misread as a (nonexistent) duration.
        meta_line = f"`{slide_id}` · ⏱ {tag}" if re.fullmatch(r"\d+:\d\d", tag) else f"`{slide_id}` · {tag}"

        out.append(f"## {num}. {title}{header_flag}")
        out.append(meta_line)
        out.append("")

        if "Establishes" in fields:
            out.append(f"**Point:** {_reflow_plain(fields['Establishes'])}")
            out.append("")

        for callout_field, emoji in (
            ("Delivery note", "🎤"),
            ("Connect-back", "🔗"),
            ("Only quantitative claims on-slide", "🔢"),
        ):
            if callout_field in fields:
                out.append(f"{emoji} **{callout_field}:** {_reflow_plain(fields[callout_field])}")
                out.append("")

        if "Cut/keep/backup decision" in fields:
            out.append(
                f"✂️ *(Phase 39 trim note — corroboration cut to anchor; "
                f"full text in parking-lot.md)*"
            )
            out.append("")

        if "Script" in fields:
            script_text = _unwrap_blockquote(fields["Script"])
            for bullet in _script_to_bullets(script_text):
                out.append(f"- {bullet}")
            out.append("")

        out.append("---")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--narrative", type=Path, required=True)
    parser.add_argument("--slides-md", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    narrative_text = args.narrative.read_text("utf-8")
    slides_md_text = args.slides_md.read_text("utf-8")
    cheatsheet = build_cheatsheet(narrative_text, slides_md_text)
    args.output.write_text(cheatsheet, "utf-8")
    print(f"Wrote {args.output} ({len(cheatsheet.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
