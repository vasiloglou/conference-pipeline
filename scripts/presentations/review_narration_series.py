#!/usr/bin/env python3
"""Series-level review for generated conference narration scripts.

Single-deck lint catches local defects. This script catches cross-deck listener
fatigue: repeated figure-detail narration, repeated chart patter, and shared
phrases that sound fine once but mechanical across a full conference series.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SLIDE_HEADER_RE = re.compile(r"(?m)^### \[(?:SLIDE|ZOOM)\s+(?P<number>\d+)\]\s*(?P<title>.+)$")
TURN_RE = re.compile(r"(?m)^\s*-\s+\*\*(?P<speaker>[^*:]+):\*\*\s*(?P<text>.+?)\s*$")
SAY_BLOCK_RE = re.compile(
    r"(?ms)^\*\*SAY:\*\*\s*(.*?)(?=^\*\*[A-Z][A-Z ]*:?\*\*|^---\s*$|\Z)"
)
WORD_RE = re.compile(r"[a-z0-9]+(?:k)?")

GENERIC_PHRASES = {
    "the key insight",
    "the core problem",
    "the big idea",
    "the important part",
    "the useful part",
    "the practical question",
    "the practical test",
    "what this shows",
    "what matters here",
    "the entire point",
    "the whole point",
    "at the conference",
    "from the conference",
    "the conference is",
    "this is the",
    "that is the",
    "it is the",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "this",
    "to",
    "up",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
}
ANCHOR_WORDS = {
    "architecture",
    "architectures",
    "cache",
    "context",
    "curve",
    "design",
    "figure",
    "flatten",
    "memory",
    "pressure",
    "scaffold",
    "scaffolds",
    "scaling",
    "target",
    "tokens",
}


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    value: Any = None
    threshold: Any = None
    examples: list[dict[str, Any]] | None = None


@dataclass
class SlideBlock:
    script_md: str
    deck_id: str
    slide_number: int
    title: str
    dialogue: str


def _deck_id(path: Path) -> str:
    return path.parent.name


def _normalize_text(text: str, conference_label: str | None = None) -> str:
    normalized = text.lower()
    if conference_label:
        normalized = re.sub(re.escape(conference_label.lower()), "the conference", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_title(title: str) -> str:
    title = re.sub(r"\s+[-—]\s+figure detail$", "", title.strip(), flags=re.I)
    title = re.sub(r"^executive summary:\s*", "", title, flags=re.I)
    title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return re.sub(r"\s+", " ", title).strip()


def _script_paths(decks_dir: Path | None, explicit: list[Path], *, script_name: str) -> list[Path]:
    paths: list[Path] = []
    if decks_dir:
        paths.extend(sorted(decks_dir.glob(f"*/{script_name}")))
    paths.extend(explicit)
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _slide_blocks(path: Path) -> list[SlideBlock]:
    text = path.read_text(encoding="utf-8")
    headers = list(SLIDE_HEADER_RE.finditer(text))
    blocks: list[SlideBlock] = []
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[start:end]
        turns = [turn.group("text").strip() for turn in TURN_RE.finditer(block)]
        if turns:
            dialogue = " ".join(turns)
        else:
            say_parts = [
                re.sub(r"\s+", " ", re.sub(r"(?m)^>\s*", "", say.group(1))).strip()
                for say in SAY_BLOCK_RE.finditer(block)
            ]
            dialogue = " ".join(part for part in say_parts if part)
        if not dialogue:
            continue
        title = match.group("title").strip()
        slide_number = int(match.group("number"))
        if slide_number < 100000 and "figure detail" not in title.lower():
            continue
        blocks.append(
            SlideBlock(
                script_md=str(path),
                deck_id=_deck_id(path),
                slide_number=slide_number,
                title=title,
                dialogue=dialogue,
            )
        )
    return blocks


def _important_phrases(text: str, conference_label: str | None) -> set[str]:
    normalized = _normalize_text(text, conference_label)
    words = WORD_RE.findall(normalized)
    phrases: set[str] = set()
    for size in (3, 4):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[index : index + size]
            phrase = " ".join(phrase_words)
            if phrase in GENERIC_PHRASES:
                continue
            if all(word in STOPWORDS for word in phrase_words):
                continue
            has_anchor = any(word in ANCHOR_WORDS or re.fullmatch(r"\d+k", word) for word in phrase_words)
            has_content = sum(1 for word in phrase_words if word not in STOPWORDS) >= 2
            if has_anchor and has_content:
                phrases.add(phrase)
    for quoted in re.findall(r"['\"]([A-Za-z0-9][A-Za-z0-9\s-]{5,60}?[A-Za-z0-9])['\"]", text):
        quoted_norm = _normalize_text(quoted, conference_label)
        quoted_words = quoted_norm.split()
        if 2 <= len(quoted_words) <= 6 and quoted_norm not in GENERIC_PHRASES:
            phrases.add(quoted_norm)
    return phrases


def _status(findings: list[Finding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "failed"
    if findings:
        return "warning"
    return "passed"


def _near_duplicate_findings(
    blocks: list[SlideBlock],
    *,
    conference_label: str | None,
    similarity_threshold: float,
    limit: int,
) -> list[Finding]:
    findings: list[Finding] = []
    by_key: dict[str, list[SlideBlock]] = defaultdict(list)
    for block in blocks:
        by_key[f"slide:{block.slide_number}"].append(block)
        title_key = _normalize_title(block.title)
        if title_key:
            by_key[f"title:{title_key}"].append(block)

    examples: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, int, int]] = set()
    for key, grouped in by_key.items():
        if len({block.deck_id for block in grouped}) < 2:
            continue
        for left_index, left in enumerate(grouped):
            for right in grouped[left_index + 1 :]:
                if left.deck_id == right.deck_id:
                    continue
                pair_key = (
                    left.deck_id,
                    right.deck_id,
                    left.slide_number,
                    right.slide_number,
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                left_text = _normalize_text(left.dialogue, conference_label)
                right_text = _normalize_text(right.dialogue, conference_label)
                if min(len(left_text), len(right_text)) < 120:
                    continue
                similarity = SequenceMatcher(None, left_text, right_text).ratio()
                if similarity >= similarity_threshold:
                    examples.append(
                        {
                            "key": key,
                            "similarity": round(similarity, 4),
                            "left": {
                                "deck_id": left.deck_id,
                                "script_md": left.script_md,
                                "slide_number": left.slide_number,
                                "title": left.title,
                            },
                            "right": {
                                "deck_id": right.deck_id,
                                "script_md": right.script_md,
                                "slide_number": right.slide_number,
                                "title": right.title,
                            },
                        }
                    )
                    if len(examples) >= limit:
                        break
            if len(examples) >= limit:
                break
        if len(examples) >= limit:
            break
    if examples:
        findings.append(
            Finding(
                rule="cross_deck_near_duplicate_figure_dialogue",
                severity="fail",
                message="Figure-detail narration is too similar across different decks.",
                value=len(examples),
                threshold=similarity_threshold,
                examples=examples,
            )
        )
    return findings


def _phrase_findings(
    blocks: list[SlideBlock],
    *,
    conference_label: str | None,
    warn_decks: int,
    fail_decks: int,
    occurrence_threshold: int,
    limit: int,
) -> list[Finding]:
    phrase_counts: Counter[str] = Counter()
    phrase_decks: dict[str, set[str]] = defaultdict(set)
    phrase_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        for phrase in _important_phrases(block.dialogue, conference_label):
            phrase_counts[phrase] += 1
            phrase_decks[phrase].add(block.deck_id)
            if len(phrase_examples[phrase]) < 4:
                phrase_examples[phrase].append(
                    {
                        "deck_id": block.deck_id,
                        "script_md": block.script_md,
                        "slide_number": block.slide_number,
                        "title": block.title,
                    }
                )

    rows: list[dict[str, Any]] = []
    for phrase, count in phrase_counts.most_common():
        deck_count = len(phrase_decks[phrase])
        if deck_count < warn_decks or count < occurrence_threshold:
            continue
        rows.append(
            {
                "phrase": phrase,
                "occurrences": count,
                "deck_count": deck_count,
                "examples": phrase_examples[phrase],
            }
        )
        if len(rows) >= limit:
            break
    if not rows:
        return []
    max_decks = max(row["deck_count"] for row in rows)
    severity = "fail" if max_decks >= fail_decks else "warn"
    return [
        Finding(
            rule="cross_deck_repeated_figure_phrase",
            severity=severity,
            message="Figure-detail narration repeats the same technical phrase across many decks.",
            value=max_decks,
            threshold={"warn_decks": warn_decks, "fail_decks": fail_decks},
            examples=rows,
        )
    ]


def review_narration_series(
    *,
    decks_dir: Path | None = None,
    script_md: list[Path] | None = None,
    script_name: str = "6_presenter_script_notebooklm_2host.md",
    conference_label: str | None = None,
    similarity_threshold: float = 0.78,
    phrase_warn_decks: int = 4,
    phrase_fail_decks: int = 8,
    phrase_occurrence_threshold: int = 6,
    example_limit: int = 20,
) -> dict[str, Any]:
    paths = _script_paths(decks_dir, script_md or [], script_name=script_name)
    blocks: list[SlideBlock] = []
    for path in paths:
        if path.exists():
            blocks.extend(_slide_blocks(path))

    findings = []
    findings.extend(
        _near_duplicate_findings(
            blocks,
            conference_label=conference_label,
            similarity_threshold=similarity_threshold,
            limit=example_limit,
        )
    )
    findings.extend(
        _phrase_findings(
            blocks,
            conference_label=conference_label,
            warn_decks=phrase_warn_decks,
            fail_decks=phrase_fail_decks,
            occurrence_threshold=phrase_occurrence_threshold,
            limit=example_limit,
        )
    )
    payload_findings = [asdict(finding) for finding in findings]
    status = _status(findings)
    return {
        "status": status,
        "approve_for_tts": status != "failed",
        "metrics": {
            "script_count": len(paths),
            "script_name": script_name,
            "figure_detail_block_count": len(blocks),
            "deck_count": len({block.deck_id for block in blocks}),
            "finding_count": len(findings),
        },
        "findings": payload_findings,
        "finding_counts": dict(Counter(finding["severity"] for finding in payload_findings)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Narration Series Review",
        "",
        f"**Status:** `{report['status']}`",
        f"**Approve for TTS:** {'yes' if report['approve_for_tts'] else 'no'}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in report["metrics"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("- None detected.")
        return "\n".join(lines).rstrip() + "\n"
    for finding in report["findings"]:
        lines.append(f"- **{finding['severity'].upper()} {finding['rule']}:** {finding['message']}")
        examples = finding.get("examples") or []
        if finding["rule"] == "cross_deck_repeated_figure_phrase":
            for row in examples[:10]:
                lines.append(
                    f"  - `{row['phrase']}`: {row['occurrences']} occurrences across "
                    f"{row['deck_count']} decks"
                )
        else:
            for row in examples[:10]:
                left = row["left"]
                right = row["right"]
                lines.append(
                    f"  - `{left['deck_id']}` slide {left['slide_number']} and "
                    f"`{right['deck_id']}` slide {right['slide_number']} "
                    f"similarity={row['similarity']}"
                )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks-dir", type=Path)
    parser.add_argument("--script-md", type=Path, action="append", default=[])
    parser.add_argument("--script-name", default="6_presenter_script_notebooklm_2host.md")
    parser.add_argument("--conference-label")
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--phrase-warn-decks", type=int, default=4)
    parser.add_argument("--phrase-fail-decks", type=int, default=8)
    parser.add_argument("--phrase-occurrence-threshold", type=int, default=6)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = review_narration_series(
        decks_dir=args.decks_dir,
        script_md=args.script_md,
        script_name=args.script_name,
        conference_label=args.conference_label,
        similarity_threshold=args.similarity_threshold,
        phrase_warn_decks=args.phrase_warn_decks,
        phrase_fail_decks=args.phrase_fail_decks,
        phrase_occurrence_threshold=args.phrase_occurrence_threshold,
        example_limit=args.example_limit,
    )
    json_payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json_payload, encoding="utf-8")
    else:
        print(json_payload, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"narration series review: {report['status']} "
        f"({report['finding_counts'].get('fail', 0)} fail, "
        f"{report['finding_counts'].get('warn', 0)} warn)",
        file=sys.stderr,
    )
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
