#!/usr/bin/env python3
"""Deterministic listener-quality lint for generated two-host narration scripts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TURN_RE = re.compile(r"(?m)^\s*-\s+\*\*(?P<speaker>[^*:]+):\*\*\s*(?P<text>.+?)\s*$")
LINE_TURN_RE = re.compile(r"^\s*-\s+\*\*(?P<speaker>[^*:]+):\*\*\s*(?P<text>.+?)\s*$")
SLIDE_RE = re.compile(r"(?m)^### \[(?:SLIDE|ZOOM)\s+(?P<number>\d+)\]\s*(?P<title>.+)$")
HOST_STAGE_RE = re.compile(r"(?m)^Host [AB]:\s+")
TRANSITION_LABEL_RE = re.compile(
    r"(?ms)^\*\*TRANSITION:\*\*\s*(.*?)(?=^\*\*[A-Z][A-Z ]*:\*\*|^---\s*$|\Z)"
)
TRANSITION_MARKER_RE = re.compile(r"(?m)^\*\*TRANSITION MARKER:\*\*\s*(.+?)\s*$")
CONFERENCE_LABEL_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\s+20\d{2}\b")
FORMULAIC_ATTENTION_RE = re.compile(r"\bWhat should (?:we|listeners) pay closest attention to here\?", re.I)
# Spoken references to WHERE a work was presented instead of WHAT it is. A
# poster always corresponds to a paper; "this poster", "poster 63306", a media
# file stem, or a transcript hash tells the audience nothing they can look up.
LOCATOR_REFERENCE_RE = re.compile(
    # ...but not the idiom "the poster child of ..." (a CVPR 2026 narration).
    r"\b(?:this|that|the|a|our|one)\s+poster\b(?!\s+(?:session|child|children)\b)"
    r"|\bposter[-\s]?\d{3,}\b"
    r"|\btalk_media_\d+"
    r"|\b(?:speech_transcript|paper_abstract)\b"
    r"|\b[0-9a-f]{12,}\b"
    r"|\bposter\s+number\b",
    re.I,
)
FORMULAIC_AGREEMENT_OPENER_RE = re.compile(
    r"(?m)^\s*-\s+\*\*[^*:]+:\*\*\s*(?P<opener>Exactly|Precisely)\.\s*",
    re.I,
)
DANGLING_FINAL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "because",
    "but",
    "if",
    "it's",
    "its",
    "or",
    "the",
    "when",
    "while",
    "without",
}
SHORT_FINAL_WORD_ALLOWLIST = {
    "ai",
    "am",
    "ar",
    "as",
    "at",
    "be",
    "by",
    "d",
    "do",
    "go",
    "he",
    "in",
    "is",
    "it",
    "k",
    "me",
    "no",
    "of",
    "ok",
    "on",
    "pm",
    "rl",
    "so",
    "to",
    "ui",
    "up",
    "us",
    "ux",
    "vr",
    "we",
    "z",
}
KNOWN_TRUNCATED_FINAL_WORDS = {"readines"}
MULTIPLE_CHOICE_END_RE = re.compile(
    r"\b[A-Z](?:\s*,\s*[A-Z])+(?:,?\s+(?:or|and)\s+[A-Z])?[.!?][\"')\]]*$"
)
ACRONYM_OR_LETTER_END_RE = re.compile(r"(?:\b[A-Z](?:[&/][A-Z])+|\b[A-Z]\+?)[.!?][\"')\]]*$")
IDIOMATIC_DANGLING_END_RE = re.compile(r"\b(?:anything but|for a while|in a while|per se)[.!?][\"')\]]*$", re.I)


THRESHOLDS = {
    "topic": {
        "yes_and_count": (10, 20),
        "exclamation_density": (8.0, 12.0),
        "question_density": (30.0, 45.0),
        "question_ending_rate": (0.20, 0.35),
        "conference_label_mentions": (12, 25),
        "formulaic_attention_question_count": (2, 6),
        "formulaic_agreement_opener_count": (4, 12),
    },
    "executive": {
        "yes_and_count": (3, 6),
        "exclamation_density": (4.0, 8.0),
        "question_density": (20.0, 35.0),
        "question_ending_rate": (0.10, 0.20),
        "conference_label_mentions": (5, 10),
        "formulaic_attention_question_count": (1, 2),
        "formulaic_agreement_opener_count": (2, 4),
    },
}


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    metric: str | None = None
    value: float | int | str | None = None
    threshold: float | int | str | None = None


def _deck_family(deck_kind: str) -> str:
    return "executive" if deck_kind.startswith("executive") else "topic"


def _clean_transition(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^>\s*", "", line.strip())
        cleaned = re.sub(r"^[-*]\s+", "", cleaned).strip()
        if cleaned:
            lines.append(cleaned)
    return re.sub(r"\s+", " ", " ".join(lines)).strip().strip('"').strip()


def count_source_transitions(source_text: str) -> int:
    return sum(1 for match in TRANSITION_LABEL_RE.finditer(source_text) if _clean_transition(match.group(1)))


def _slide_blocks(text: str) -> list[str]:
    headers = list(SLIDE_RE.finditer(text))
    blocks: list[str] = []
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        blocks.append(text[start:end])
    return blocks


def _turn_texts(text: str) -> list[str]:
    return [match.group("text").strip() for match in TURN_RE.finditer(text)]


def _dialogue_line_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = LINE_TURN_RE.match(line)
        if not match:
            continue
        items.append(
            {
                "line": line_number,
                "speaker": match.group("speaker").strip(),
                "text": match.group("text").strip(),
            }
        )
    return items


def _final_question_count(text: str) -> int:
    count = 0
    for block in _slide_blocks(text):
        turns = _turn_texts(block)
        if turns and turns[-1].rstrip().endswith("?"):
            count += 1
    return count


def _conference_mentions(text: str, conference_label: str | None) -> int:
    if conference_label:
        return len(re.findall(re.escape(conference_label), text, flags=re.I))
    return len(CONFERENCE_LABEL_RE.findall(text))


def _final_word(text: str) -> str:
    match = re.search(r"([A-Za-z][A-Za-z'’-]*)[.!?][\"')\]]*$", text.strip())
    if not match:
        return ""
    return match.group(1).replace("’", "'").lower()


def _raw_final_word(text: str) -> str:
    match = re.search(r"([A-Za-z][A-Za-z'’-]*)[.!?][\"')\]]*$", text.strip())
    if not match:
        return ""
    return match.group(1).replace("’", "'")


def _has_unbalanced_quote_boundary(text: str) -> bool:
    if text.count('"') % 2:
        return True
    return False


def _looks_truncated_or_hanging(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _has_unbalanced_quote_boundary(stripped):
        return True
    if re.search(r"\bet\s+al\.$", stripped, flags=re.I):
        return True
    if MULTIPLE_CHOICE_END_RE.search(stripped):
        return False
    if ACRONYM_OR_LETTER_END_RE.search(stripped):
        return False
    if IDIOMATIC_DANGLING_END_RE.search(stripped):
        return False
    raw_final_word = _raw_final_word(stripped)
    final_word = _final_word(stripped)
    if not final_word:
        return False
    if final_word in KNOWN_TRUNCATED_FINAL_WORDS:
        return True
    if final_word in DANGLING_FINAL_WORDS and not raw_final_word.isupper():
        return True
    if (
        len(final_word) <= 2
        and final_word not in SHORT_FINAL_WORD_ALLOWLIST
        and not raw_final_word.isupper()
    ):
        return True
    return False


def _truncated_or_hanging_examples(text: str, *, limit: int = 8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for item in _dialogue_line_items(text):
        if _looks_truncated_or_hanging(str(item["text"])):
            examples.append(item)
            if len(examples) >= limit:
                break
    return examples


def _status(findings: list[Finding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "failed"
    if any(finding.severity == "warn" for finding in findings):
        return "warning"
    return "passed"


def _add_threshold_finding(
    findings: list[Finding],
    *,
    rule: str,
    metric: str,
    value: float | int,
    warn: float | int,
    fail: float | int,
    message: str,
) -> None:
    if value > fail:
        findings.append(
            Finding(rule=rule, severity="fail", message=message, metric=metric, value=value, threshold=fail)
        )
    elif value > warn:
        findings.append(
            Finding(rule=rule, severity="warn", message=message, metric=metric, value=value, threshold=warn)
        )


def analyze_narration_script(
    *,
    script_md: Path,
    source_script_md: Path | None = None,
    deck_kind: str = "topic",
    topic_id: str | None = None,
    conference_label: str | None = None,
) -> dict[str, Any]:
    text = script_md.read_text(encoding="utf-8")
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    line_count = max(1, len(nonempty_lines))
    turns = list(TURN_RE.finditer(text))
    speakers = sorted({match.group("speaker").strip() for match in turns})
    slide_count = len(_slide_blocks(text))
    final_question_count = _final_question_count(text)
    source_transition_count = (
        count_source_transitions(source_script_md.read_text(encoding="utf-8"))
        if source_script_md and source_script_md.exists()
        else 0
    )
    transition_marker_count = len(TRANSITION_MARKER_RE.findall(text))
    conference_mentions = _conference_mentions(text, conference_label)
    formulaic_attention_count = len(FORMULAIC_ATTENTION_RE.findall(text))
    locator_reference_lines = [
        f"{index}: {line.strip()[:120]}"
        for index, line in enumerate(text.splitlines(), start=1)
        if LOCATOR_REFERENCE_RE.search(line)
    ]
    agreement_openers = [
        match.group("opener").lower() for match in FORMULAIC_AGREEMENT_OPENER_RE.finditer(text)
    ]
    agreement_opener_count = len(agreement_openers)
    exact_opener_count = sum(1 for opener in agreement_openers if opener == "exactly")
    precise_opener_count = sum(1 for opener in agreement_openers if opener == "precisely")
    suspicious_fragments = _truncated_or_hanging_examples(text)
    expected_executive_transition_count = (
        max(0, slide_count - 1) if _deck_family(deck_kind) == "executive" else 0
    )
    metrics: dict[str, Any] = {
        "script_md": str(script_md),
        "source_script_md": str(source_script_md) if source_script_md else None,
        "deck_kind": deck_kind,
        "topic_id": topic_id,
        "slide_count": slide_count,
        "turn_count": len(turns),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "nonempty_line_count": line_count,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "exclamation_density_per_100_lines": round(text.count("!") / line_count * 100, 2),
        "question_density_per_100_lines": round(text.count("?") / line_count * 100, 2),
        "question_ending_count": final_question_count,
        "question_ending_rate": round(final_question_count / max(1, slide_count), 4),
        "yes_and_count": len(re.findall(r"\bYes,\s+and\b", text, flags=re.I)),
        "okay_count": len(re.findall(r"\bOkay\b", text, flags=re.I)),
        "conference_label_mentions": conference_mentions,
        "formulaic_attention_question_count": formulaic_attention_count,
        "formulaic_agreement_opener_count": agreement_opener_count,
        "exactly_opener_count": exact_opener_count,
        "precisely_opener_count": precise_opener_count,
        "truncated_or_hanging_fragment_count": len(suspicious_fragments),
        "truncated_or_hanging_fragment_examples": suspicious_fragments,
        "source_transition_count": source_transition_count,
        "transition_marker_count": transition_marker_count,
        "expected_executive_transition_count": expected_executive_transition_count,
    }

    metrics["locator_reference_count"] = len(locator_reference_lines)

    findings: list[Finding] = []
    from conference_pipeline.presentation.claim_fidelity import absence_claims, closed_author_lists
    for hit in absence_claims(text):
        findings.append(Finding(rule="absence_claim", severity="fail", message=hit))
    for hit in closed_author_lists(text):
        findings.append(Finding(rule="closed_author_list", severity="fail", message=hit))
    if locator_reference_lines:
        findings.append(
            Finding(
                rule="locator_reference",
                severity="fail",
                message=(
                    "Narration refers to a source by locator instead of by work title "
                    "(a poster is a paper: name it); first example line "
                    f"{locator_reference_lines[0]}"
                ),
                metric="locator_reference_count",
                value=len(locator_reference_lines),
                threshold=0,
            )
        )
    if len(speakers) < 2:
        findings.append(
            Finding(
                rule="missing_host_markers",
                severity="fail",
                message="Generated script has fewer than two host dialogue markers.",
                metric="speaker_count",
                value=len(speakers),
                threshold=2,
            )
        )
    if HOST_STAGE_RE.search(text):
        findings.append(
            Finding(
                rule="host_stage_directions",
                severity="fail",
                message="Generated script still contains Host A/B stage directions.",
            )
        )
    if suspicious_fragments:
        first = suspicious_fragments[0]
        findings.append(
            Finding(
                rule="truncated_or_hanging_dialogue",
                severity="fail",
                message=(
                    "Dialogue contains likely truncated or hanging spoken fragments; "
                    f"first example line {first['line']}: {first['text']}"
                ),
                metric="truncated_or_hanging_fragment_count",
                value=len(suspicious_fragments),
                threshold=0,
            )
        )
    if source_transition_count and transition_marker_count == 0:
        findings.append(
            Finding(
                rule="dropped_transitions",
                severity="fail",
                message="Source transitions exist but generated script preserved no transition markers.",
                metric="transition_marker_count",
                value=0,
                threshold="> 0 when source transitions exist",
            )
        )
    elif source_transition_count and transition_marker_count / source_transition_count < 0.8:
        findings.append(
            Finding(
                rule="transition_preservation_ratio",
                severity="warn",
                message="Generated script preserved fewer than 80 percent of source transitions.",
                metric="transition_marker_count/source_transition_count",
                value=round(transition_marker_count / source_transition_count, 4),
                threshold=0.8,
            )
        )
    if expected_executive_transition_count and transition_marker_count < expected_executive_transition_count:
        findings.append(
            Finding(
                rule="missing_executive_transition_markers",
                severity="fail",
                message=(
                    "Executive summary scripts need explicit slide-to-slide transition markers "
                    "before TTS review."
                ),
                metric="transition_marker_count",
                value=transition_marker_count,
                threshold=expected_executive_transition_count,
            )
        )

    thresholds = THRESHOLDS[_deck_family(deck_kind)]
    _add_threshold_finding(
        findings,
        rule="yes_and_overuse",
        metric="yes_and_count",
        value=metrics["yes_and_count"],
        warn=thresholds["yes_and_count"][0],
        fail=thresholds["yes_and_count"][1],
        message='Formulaic "Yes, and" handoffs are overused.',
    )
    _add_threshold_finding(
        findings,
        rule="exclamation_density",
        metric="exclamation_density_per_100_lines",
        value=metrics["exclamation_density_per_100_lines"],
        warn=thresholds["exclamation_density"][0],
        fail=thresholds["exclamation_density"][1],
        message="Exclamation density is high enough to make narration feel hyped.",
    )
    _add_threshold_finding(
        findings,
        rule="question_density",
        metric="question_density_per_100_lines",
        value=metrics["question_density_per_100_lines"],
        warn=thresholds["question_density"][0],
        fail=thresholds["question_density"][1],
        message="Question density makes the narration feel interrogative rather than guided.",
    )
    _add_threshold_finding(
        findings,
        rule="question_ending_rate",
        metric="question_ending_rate",
        value=metrics["question_ending_rate"],
        warn=thresholds["question_ending_rate"][0],
        fail=thresholds["question_ending_rate"][1],
        message="Too many slide-final turns end as question hooks instead of observations.",
    )
    if conference_mentions > thresholds["conference_label_mentions"][0]:
        findings.append(
            Finding(
                rule="conference_label_repetition",
                severity="warn",
                message="Conference label repetition may sound like spoken filler.",
                metric="conference_label_mentions",
                value=conference_mentions,
                threshold=thresholds["conference_label_mentions"][0],
            )
        )
    _add_threshold_finding(
        findings,
        rule="formulaic_attention_question",
        metric="formulaic_attention_question_count",
        value=formulaic_attention_count,
        warn=thresholds["formulaic_attention_question_count"][0],
        fail=thresholds["formulaic_attention_question_count"][1],
        message='The repeated "pay closest attention" question makes the dialogue mechanical.',
    )
    _add_threshold_finding(
        findings,
        rule="formulaic_agreement_opener",
        metric="formulaic_agreement_opener_count",
        value=agreement_opener_count,
        warn=thresholds["formulaic_agreement_opener_count"][0],
        fail=thresholds["formulaic_agreement_opener_count"][1],
        message='Repeated "Exactly." / "Precisely." turn openers make the hosts sound scripted.',
    )

    findings_payload = [asdict(finding) for finding in findings]
    return {
        "status": _status(findings),
        "metrics": metrics,
        "findings": findings_payload,
        "finding_counts": dict(Counter(finding["severity"] for finding in findings_payload)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-md", type=Path, required=True)
    parser.add_argument("--source-script-md", type=Path)
    parser.add_argument(
        "--deck-kind",
        default="topic",
        choices=["topic", "executive", "executive_short", "executive_long"],
    )
    parser.add_argument("--topic-id")
    parser.add_argument("--conference-label")
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_narration_script(
        script_md=args.script_md,
        source_script_md=args.source_script_md,
        deck_kind=args.deck_kind,
        topic_id=args.topic_id,
        conference_label=args.conference_label,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(
        f"narration lint: {report['status']} "
        f"({report['finding_counts'].get('fail', 0)} fail, "
        f"{report['finding_counts'].get('warn', 0)} warn)",
        file=sys.stderr,
    )
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
