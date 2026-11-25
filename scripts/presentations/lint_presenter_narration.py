#!/usr/bin/env python3
"""Deterministic pre-TTS lint for single-presenter narration scripts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_presenter_video import (  # noqa: E402
    FORBIDDEN_TTS_TEXT_PATTERNS,
    SlideNarration,
    normalize_space,
    parse_presenter_script,
)


WELCOME_RE = re.compile(
    r"^(?:welcome(?:,?\s+everyone| everyone)?(?:\s+to\s+[^.]+)?|"
    r"good morning, everyone(?:,?\s+and welcome)?|"
    r"hello everyone(?:,?\s+and welcome)?)\b",
    re.I,
)
CLOSING_RE = re.compile(
    r"^(?:looking (?:ahead|forward)|where do we go from here|our roadmap(?: forward)? is|the path forward is clear)\b",
    re.I,
)
CONFERENCE_LABEL_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\s+20\d{2}\b")
FILLER_TERMS = ("critical", "crucial", "key", "fundamental")
STITCH_TEMPLATE_PHRASES = (
    "That connects directly to",
    "The operational test is",
    "That sets up",
)
IMPERATIVE_GLUE_RE = re.compile(
    r"\b(?:That connects directly to|The operational test is)\s+"
    r"(?:measure|treat|prioritize|fund|require|implement|maintain|gate|join|steward)\b",
    re.I,
)
CLOSING_GRAMMAR_GLUE_RE = re.compile(
    r"\b(?:The next research phase is We need|The next step is We need|"
    r"The hard part now is First, we need|The hard part now is We need|"
    r"The unfinished business is We need)\b",
    re.I,
)


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    metric: str | None = None
    value: float | int | str | None = None
    threshold: float | int | str | None = None


def _status(findings: list[Finding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "failed"
    if any(finding.severity == "warn" for finding in findings):
        return "warning"
    return "passed"


def _entry_text(entries: list[SlideNarration]) -> str:
    return "\n".join(entry.say for entry in entries)


def _load_segments(script_json: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not script_json:
        return [], []
    payload = json.loads(script_json.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    slides = payload.get("metadata", {}).get("slides", [])
    return (
        segments if isinstance(segments, list) else [],
        slides if isinstance(slides, list) else [],
    )


def _conference_mentions(text: str, conference_label: str | None) -> int:
    if conference_label:
        return len(re.findall(re.escape(conference_label), text, flags=re.I))
    return len(CONFERENCE_LABEL_RE.findall(text))


def _transition_preservation(entries: list[SlideNarration], segment_texts: list[str]) -> tuple[int, int]:
    expected = [entry.transition for entry in entries if entry.transition]
    found = 0
    haystack = "\n".join(normalize_space(text).lower() for text in segment_texts)
    for transition in expected:
        normalized = normalize_space(transition or "").strip().strip('"').lower()
        if normalized and normalized in haystack:
            found += 1
    return len(expected), found


def analyze_presenter_narration(
    *,
    script_md: Path,
    script_json: Path | None = None,
    deck_kind: str = "topic",
    topic_id: str | None = None,
    conference_label: str | None = None,
) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        entries = parse_presenter_script(script_md, max_context_items=99, context_prefix="For context")
    except SystemExit as exc:
        entries = []
        findings.append(
            Finding(
                rule="presenter_script_parse_failed",
                severity="fail",
                message=str(exc),
            )
        )

    segments, metadata_slides = _load_segments(script_json)
    segment_texts = [str(segment.get("text") or "") for segment in segments if isinstance(segment, dict)]
    all_tts_text = "\n".join(segment_texts)
    say_text = _entry_text(entries)
    from conference_pipeline.presentation.claim_fidelity import absence_claims, closed_author_lists
    for hit in absence_claims(script_md.read_text(encoding="utf-8") + "\n" + all_tts_text):
        findings.append(Finding(rule="absence_claim", severity="fail", message=hit))
    for hit in closed_author_lists(script_md.read_text(encoding="utf-8") + "\n" + all_tts_text):
        findings.append(Finding(rule="closed_author_list", severity="fail", message=hit))
    expected_transitions, preserved_transitions = _transition_preservation(entries, segment_texts)
    filler_counts = {
        term: len(re.findall(rf"\b{re.escape(term)}\b", say_text, flags=re.I))
        for term in FILLER_TERMS
    }
    stitch_counts = {
        phrase: len(re.findall(re.escape(phrase), say_text, flags=re.I))
        for phrase in STITCH_TEMPLATE_PHRASES
    }
    stitch_transition_counts = {
        phrase: sum(
            len(re.findall(re.escape(phrase), entry.transition or "", flags=re.I))
            for entry in entries
        )
        for phrase in STITCH_TEMPLATE_PHRASES
    }
    imperative_glue_count = len(IMPERATIVE_GLUE_RE.findall(say_text))
    closing_grammar_glue_count = len(CLOSING_GRAMMAR_GLUE_RE.findall(say_text))

    metrics: dict[str, Any] = {
        "script_md": str(script_md),
        "script_json": str(script_json) if script_json else None,
        "deck_kind": deck_kind,
        "topic_id": topic_id,
        "slide_count": len(entries),
        "segment_count": len(segment_texts),
        "metadata_slide_count": len(metadata_slides),
        "say_word_count": len(re.findall(r"\b\w+\b", say_text)),
        "transition_count": expected_transitions,
        "transition_preserved_count": preserved_transitions,
        "conference_label_mentions": _conference_mentions(all_tts_text, conference_label),
        "welcome_formula_opening": bool(entries and WELCOME_RE.search(entries[0].say.strip())),
        "closing_formula_opening": bool(entries and CLOSING_RE.search(entries[-1].say.strip())),
        "filler_term_counts": filler_counts,
        "stitch_template_counts": stitch_counts,
        "stitch_transition_counts": stitch_transition_counts,
        "imperative_glue_count": imperative_glue_count,
        "closing_grammar_glue_count": closing_grammar_glue_count,
    }

    if not entries:
        findings.append(
            Finding(
                rule="missing_presenter_say_blocks",
                severity="fail",
                message="Presenter script has no parseable spoken SAY blocks.",
                metric="slide_count",
                value=0,
                threshold="> 0",
            )
        )

    if script_json:
        if len(segment_texts) != len(entries):
            findings.append(
                Finding(
                    rule="script_json_segment_mismatch",
                    severity="fail",
                    message="script.json segment count does not match parsed presenter narration entries.",
                    metric="segment_count/slide_count",
                    value=f"{len(segment_texts)}/{len(entries)}",
                    threshold="equal",
                )
            )
        for index, text in enumerate(segment_texts, start=1):
            for rule, pattern in FORBIDDEN_TTS_TEXT_PATTERNS:
                if pattern.search(text):
                    findings.append(
                        Finding(
                            rule=f"tts_forbidden_{rule}",
                            severity="fail",
                            message=f"script.json segment {index} contains non-spoken text matching {rule}.",
                            metric="segment",
                            value=index,
                        )
                    )
                    break

    if expected_transitions and preserved_transitions == 0:
        findings.append(
            Finding(
                rule="dropped_presenter_transitions",
                severity="fail",
                message="Presenter script has transitions but script.json preserves none of them.",
                metric="transition_preserved_count",
                value=0,
                threshold="> 0",
            )
        )
    elif expected_transitions and preserved_transitions / expected_transitions < 0.8:
        findings.append(
            Finding(
                rule="presenter_transition_preservation_ratio",
                severity="warn",
                message="script.json preserves fewer than 80 percent of presenter transitions.",
                metric="transition_preserved_count/transition_count",
                value=round(preserved_transitions / expected_transitions, 4),
                threshold=0.8,
            )
        )

    if metrics["welcome_formula_opening"]:
        findings.append(
            Finding(
                rule="welcome_formula_opening",
                severity="warn",
                message="Opening SAY starts with a reusable welcome formula instead of the topic stake.",
            )
        )
    if metrics["closing_formula_opening"]:
        findings.append(
            Finding(
                rule="closing_formula_opening",
                severity="warn",
                message="Closing SAY starts with a repeated roadmap/looking-ahead formula.",
            )
        )
    for term, count in filler_counts.items():
        if count > 5:
            findings.append(
                Finding(
                    rule=f"filler_{term}_overuse",
                    severity="warn",
                    message=f'The word "{term}" appears often enough to sound like filler.',
                    metric=f"{term}_count",
                    value=count,
                    threshold=5,
                )
            )

    stitch_total = sum(stitch_counts.values()) + sum(stitch_transition_counts.values())
    stitch_limit = 4 if deck_kind.startswith("executive") else 10
    if stitch_total > stitch_limit:
        findings.append(
            Finding(
                rule="stitch_template_repetition",
                severity="fail" if deck_kind.startswith("executive") else "warn",
                message=(
                    "Presenter narration repeats assembly-template phrases such as "
                    "\"That connects directly to\", \"The operational test is\", or \"That sets up\"."
                ),
                metric="stitch_template_total",
                value=stitch_total,
                threshold=stitch_limit,
            )
        )
    if imperative_glue_count:
        findings.append(
            Finding(
                rule="imperative_glue_template",
                severity="fail" if deck_kind.startswith("executive") else "warn",
                message="Presenter narration glues imperative bullets to a preposition, producing mechanical grammar.",
                metric="imperative_glue_count",
                value=imperative_glue_count,
                threshold=0,
            )
        )
    if closing_grammar_glue_count:
        findings.append(
            Finding(
                rule="closing_grammar_glue",
                severity="fail",
                message='Closing narration contains an audible "is We need..." grammar join.',
                metric="closing_grammar_glue_count",
                value=closing_grammar_glue_count,
                threshold=0,
            )
        )

    if metrics["conference_label_mentions"] > (10 if deck_kind.startswith("executive") else 25):
        findings.append(
            Finding(
                rule="conference_label_repetition",
                severity="warn",
                message="Conference label repetition may sound like spoken filler.",
                metric="conference_label_mentions",
                value=metrics["conference_label_mentions"],
            )
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
    parser.add_argument("--script-json", type=Path)
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
    report = analyze_presenter_narration(
        script_md=args.script_md,
        script_json=args.script_json,
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
        f"presenter narration lint: {report['status']} "
        f"({report['finding_counts'].get('fail', 0)} fail, "
        f"{report['finding_counts'].get('warn', 0)} warn)",
        file=sys.stderr,
    )
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
