#!/usr/bin/env python3
"""Deterministic repair/sync helper for single-presenter narration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_presenter_video import parse_presenter_script, write_script_json  # noqa: E402


SAY_BLOCK_RE = re.compile(
    r"(?ms)(?P<label>^\*\*SAY:\*\*\s*\n)(?P<body>.*?)(?=^\*\*[A-Z][A-Z ]*:?\*\*|^---\s*$|\Z)"
)
WELCOME_PREFIX_RE = re.compile(
    r"^(?P<prefix>\s*>\s*)?(?:"
    r"welcome(?:,?\s+everyone| everyone)?(?:\s+to\s+[^.]+)?\.|"
    r"good morning, everyone(?:,?\s+and welcome)?\.|"
    r"hello everyone(?:,?\s+and welcome)?\.)\s+",
    re.I,
)
CLOSING_REPLACEMENTS = (
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Looking ahead,\s+our roadmap is focused on\s*", re.I),
        "The next priorities are ",
    ),
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Looking ahead,\s+the ultimate vision is\s*", re.I),
        "The larger goal is ",
    ),
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Looking ahead,\s+the next\s+3[–-]5 years will be about\s*", re.I),
        "The next three to five years should focus on ",
    ),
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Looking ahead,\s+the path is challenging but clear[:.]?\s*", re.I),
        "The hard part now is ",
    ),
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Looking forward,\s+the roadmap for\s+(.+?)\s+is clear[:.]?\s*", re.I),
        "The next research phase is ",
    ),
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Looking forward,\s+the trend is clear[:.]?\s*", re.I),
        "The trend now points to ",
    ),
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Where do we go from here\?\s+The roadmap is clear[:.]?\s*", re.I),
        "The roadmap now turns on ",
    ),
    (
        re.compile(r"^(?P<prefix>\s*>\s*)?Where do we go from here\?\s+The path forward is clear[:.]?\s*", re.I),
        "The next step is ",
    ),
    (re.compile(r"^(?P<prefix>\s*>\s*)?Looking ahead,\s*", re.I), "The next milestone is "),
    (re.compile(r"^(?P<prefix>\s*>\s*)?Looking forward,\s*", re.I), "The next milestone is "),
    (re.compile(r"^(?P<prefix>\s*>\s*)?Where do we go from here\?\s*", re.I), "The unfinished business is "),
    (re.compile(r"^(?P<prefix>\s*>\s*)?Our roadmap(?: forward)? is focused on\s*", re.I), "The roadmap focuses on "),
    (re.compile(r"^(?P<prefix>\s*>\s*)?Our roadmap(?: forward)? is\s*", re.I), "The roadmap is "),
)
STITCH_REPLACEMENTS = (
    (
        re.compile(
            r"\bThat connects directly to\s+"
            r"(?P<verb>measure|treat|prioritize|fund|require|implement|maintain|gate|join|steward)\b",
            re.I,
        ),
        "That also means",
        "gerund",
    ),
    (
        re.compile(
            r"\bThe operational test is\s+"
            r"(?P<verb>measure|treat|prioritize|fund|require|implement|maintain|gate|join|steward)\b",
            re.I,
        ),
        "The operational test is whether teams can",
        "plain",
    ),
    (
        re.compile(r"\bThat connects directly to\s+", re.I),
        "The next implication is that ",
        "literal",
    ),
    (
        re.compile(r"\bThe operational test is\s+metrics must include\b", re.I),
        "The operating standard is that metrics include",
        "literal",
    ),
)
TRANSITION_STITCH_RE = re.compile(r'(?P<prefix>\*\*TRANSITION:\*\*\s*")That sets up\s+(?P<title>[^"]+?)\.?(?P<suffix>")')
GRAMMAR_GLUE_REPLACEMENTS = (
    (
        re.compile(r"\bThe next research phase is We need\b", re.I),
        "The next research phase needs",
    ),
    (
        re.compile(r"\bThe next step is We need to\b", re.I),
        "The next step is to",
    ),
    (
        re.compile(r"\bThe hard part now is First, we need to\b", re.I),
        "The hard part now is to first",
    ),
    (
        re.compile(r"\bThe hard part now is We need to\b", re.I),
        "The hard part now is to",
    ),
    (
        re.compile(r"\bThe unfinished business is We need to\b", re.I),
        "The unfinished business is to",
    ),
)
VERB_GERUNDS = {
    "fund": "funding",
    "gate": "gating",
    "implement": "implementing",
    "join": "joining",
    "maintain": "maintaining",
    "measure": "measuring",
    "prioritize": "prioritizing",
    "require": "requiring",
    "steward": "stewarding",
    "treat": "treating",
}
FILLER_REPLACEMENTS = {
    "critical": ("important", "central", "material", "decisive"),
    "crucial": ("important", "essential", "central"),
    "key": ("main", "specific", "practical", "central"),
    "fundamental": ("foundational", "underlying", "basic"),
}
FILLER_RE = re.compile(r"\b(critical|crucial|key|fundamental)\b", re.I)


@dataclass
class RepairStats:
    welcome_openers_removed: int = 0
    closing_openers_rewritten: int = 0
    stitch_templates_rewritten: int = 0
    transition_stitches_rewritten: int = 0
    grammar_glue_rewritten: int = 0
    filler_terms_replaced: int = 0
    script_json_regenerated: bool = False


def _rewrite_first_say(body: str, stats: RepairStats) -> str:
    rewritten, count = WELCOME_PREFIX_RE.subn(lambda match: match.group("prefix") or "", body, count=1)
    stats.welcome_openers_removed += count
    return rewritten


def _rewrite_last_say(body: str, stats: RepairStats) -> str:
    rewritten = body
    for pattern, replacement in CLOSING_REPLACEMENTS:
        def replace(match: re.Match[str]) -> str:
            return f"{match.group('prefix') or ''}{replacement}"

        rewritten, count = pattern.subn(replace, rewritten, count=1)
        if count:
            stats.closing_openers_rewritten += count
            break
    return rewritten


def _rewrite_stitch_templates(body: str, stats: RepairStats) -> str:
    rewritten = body
    for pattern, replacement, mode in STITCH_REPLACEMENTS:
        def replace(match: re.Match[str]) -> str:
            if mode == "gerund":
                verb = match.group("verb").lower()
                return f"{replacement} {VERB_GERUNDS.get(verb, verb + 'ing')}"
            if mode == "plain":
                return f"{replacement} {match.group('verb').lower()}"
            return replacement

        rewritten, count = pattern.subn(replace, rewritten)
        stats.stitch_templates_rewritten += count
    return rewritten


def _rewrite_grammar_glue(body: str, stats: RepairStats) -> str:
    rewritten = body
    for pattern, replacement in GRAMMAR_GLUE_REPLACEMENTS:
        rewritten, count = pattern.subn(replacement, rewritten)
        stats.grammar_glue_rewritten += count
    return rewritten


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _thin_filler_terms(
    body: str,
    counts: Counter[str],
    stats: RepairStats,
    *,
    filler_limit: int,
) -> str:
    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        term = original.lower()
        counts[term] += 1
        if counts[term] <= filler_limit:
            return original
        options = FILLER_REPLACEMENTS[term]
        replacement = options[(counts[term] - filler_limit - 1) % len(options)]
        stats.filler_terms_replaced += 1
        return _preserve_case(original, replacement)

    return FILLER_RE.sub(replace, body)


def _rewrite_transition_stitches(text: str, stats: RepairStats) -> str:
    def replace(match: re.Match[str]) -> str:
        stats.transition_stitches_rewritten += 1
        return f"{match.group('prefix')}Next, {match.group('title')}.{match.group('suffix')}"

    return TRANSITION_STITCH_RE.sub(replace, text)


def repair_markdown(text: str, stats: RepairStats, *, filler_limit: int = 5) -> str:
    matches = list(SAY_BLOCK_RE.finditer(text))
    if not matches:
        return _rewrite_transition_stitches(text, stats)
    pieces: list[str] = []
    position = 0
    filler_counts: Counter[str] = Counter()
    for index, match in enumerate(matches):
        pieces.append(text[position : match.start()])
        body = match.group("body")
        if index == 0:
            body = _rewrite_first_say(body, stats)
        if index == len(matches) - 1:
            body = _rewrite_last_say(body, stats)
        body = _rewrite_stitch_templates(body, stats)
        body = _rewrite_grammar_glue(body, stats)
        body = _thin_filler_terms(body, filler_counts, stats, filler_limit=filler_limit)
        pieces.append(match.group("label") + body)
        position = match.end()
    pieces.append(text[position:])
    return _rewrite_transition_stitches("".join(pieces), stats)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-md", type=Path, required=True)
    parser.add_argument("--script-json", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--filler-limit", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = RepairStats()
    original = args.script_md.read_text(encoding="utf-8")
    repaired = repair_markdown(original, stats, filler_limit=args.filler_limit)
    if not args.dry_run and repaired != original:
        args.script_md.write_text(repaired, encoding="utf-8")

    if args.script_json:
        entries = parse_presenter_script(args.script_md, max_context_items=99, context_prefix="For context")
        if not args.dry_run:
            write_script_json(
                entries,
                args.script_json,
                title=args.title or args.script_md.stem.replace("_", " ").title(),
            )
        stats.script_json_regenerated = True

    report = {
        "script_md": str(args.script_md),
        "script_json": str(args.script_json) if args.script_json else None,
        "dry_run": args.dry_run,
        "stats": stats.__dict__,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(
        "presenter narration fix: "
        f"{stats.welcome_openers_removed} welcome openers, "
        f"{stats.closing_openers_rewritten} closing openers, "
        f"{stats.stitch_templates_rewritten} stitch templates, "
        f"{stats.transition_stitches_rewritten} transition stitches, "
        f"{stats.grammar_glue_rewritten} grammar glue, "
        f"{stats.filler_terms_replaced} filler terms, "
        f"script_json_regenerated={stats.script_json_regenerated}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
