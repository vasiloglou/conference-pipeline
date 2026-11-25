#!/usr/bin/env python3
"""Editorial narration review gate for generated two-host scripts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lint_narration_script import analyze_narration_script  # noqa: E402


HYPE_TERMS = [
    "game-changer",
    "game changer",
    "huge shift",
    "massive shift",
    "paradigm",
    "unlock",
    "mind-blowing",
    "wild",
    "blew my mind",
    "fascinating",
]


def _load_lint_report(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _term_count(text: str, terms: list[str]) -> int:
    return sum(len(re.findall(re.escape(term), text, flags=re.I)) for term in terms)


def _score_from_metrics(report: dict[str, Any], *, hype_count: int) -> dict[str, int]:
    metrics = report["metrics"]
    findings = {finding["rule"]: finding for finding in report["findings"]}
    factual_fidelity = 5
    if {
        "host_stage_directions",
        "missing_host_markers",
        "truncated_or_hanging_dialogue",
    } & findings.keys():
        factual_fidelity = 1
    elif "dropped_transitions" in findings:
        factual_fidelity = 2
    elif "transition_preservation_ratio" in findings:
        factual_fidelity = 3

    cohesion = 5
    if "dropped_transitions" in findings or "missing_executive_transition_markers" in findings:
        cohesion = 2
    elif "transition_preservation_ratio" in findings:
        cohesion = 3
    elif metrics.get("question_ending_rate", 0) > 0.2:
        cohesion = 4

    listener_interest = 5
    if hype_count > 40:
        listener_interest = 2
    elif hype_count > 25:
        listener_interest = 3
    elif hype_count > 15:
        listener_interest = 4

    fluff_control = 5
    if hype_count > 40 or "exclamation_density" in findings:
        fluff_control = 2
    elif hype_count > 25:
        fluff_control = 3
    elif hype_count > 15:
        fluff_control = 4

    transition_quality = 5
    if "dropped_transitions" in findings or "missing_executive_transition_markers" in findings:
        transition_quality = 1
    elif "transition_preservation_ratio" in findings:
        transition_quality = 3

    naturalness = 5
    if (
        "yes_and_overuse" in findings
        or "formulaic_attention_question" in findings
        or "formulaic_agreement_opener" in findings
        or metrics.get("okay_count", 0) > 20
    ):
        naturalness = 2
    elif metrics.get("okay_count", 0) > 10:
        naturalness = 3

    pleasantness = 5
    listener_rules = {
        "question_density",
        "question_ending_rate",
        "exclamation_density",
        "yes_and_overuse",
        "formulaic_attention_question",
        "formulaic_agreement_opener",
    }
    listener_hits = listener_rules & findings.keys()
    if len(listener_hits) >= 3:
        pleasantness = 2
    elif listener_hits:
        pleasantness = 3
    elif metrics.get("conference_label_mentions", 0) > 12:
        pleasantness = 4

    return {
        "factual_fidelity": factual_fidelity,
        "cohesion": cohesion,
        "listener_interest": listener_interest,
        "fluff_control": fluff_control,
        "transition_quality": transition_quality,
        "naturalness": naturalness,
        "pleasantness": pleasantness,
    }


def _revision_brief(report: dict[str, Any], *, hype_count: int) -> list[str]:
    rules = {finding["rule"] for finding in report["findings"]}
    brief: list[str] = []
    if "truncated_or_hanging_dialogue" in rules:
        brief.append("Fix truncated or hanging dialogue fragments before any TTS render.")
    if "dropped_transitions" in rules or "transition_preservation_ratio" in rules:
        brief.append("Preserve and audibly use source transitions before rewriting dialogue.")
    if "missing_executive_transition_markers" in rules:
        brief.append("Add explicit executive summary bridge markers between adjacent slides.")
    if "question_ending_rate" in rules:
        brief.append("Rewrite slide-final turns as assertions, observations, implications, or bridge statements.")
    if "question_density" in rules:
        brief.append("Move questions into the middle of segments and answer each one immediately.")
    if "yes_and_overuse" in rules:
        brief.append('Replace repeated "Yes, and" handoffs with content-specific callbacks.')
    if "formulaic_attention_question" in rules:
        brief.append('Replace repeated "What should we pay closest attention to here?" prompts with concrete observations.')
    if "formulaic_agreement_opener" in rules:
        brief.append('Vary handoffs; avoid repeated "Exactly." and "Precisely." turn openers.')
    if "exclamation_density" in rules or hype_count > 5:
        brief.append("Reduce hype words and exclamation-driven energy; surface concrete stakes instead.")
    if "conference_label_repetition" in rules:
        brief.append("Mention the conference label only where it orients the listener.")
    if not brief:
        brief.append("No blocking rewrite needed; keep review focused on factual spot checks and pacing.")
    return brief


def review_narration_script(
    *,
    script_md: Path,
    source_script_md: Path | None = None,
    lint_report: Path | None = None,
    deck_kind: str = "topic",
    topic_id: str | None = None,
    conference_label: str | None = None,
) -> dict[str, Any]:
    report = _load_lint_report(lint_report) or analyze_narration_script(
        script_md=script_md,
        source_script_md=source_script_md,
        deck_kind=deck_kind,
        topic_id=topic_id,
        conference_label=conference_label,
    )
    text = script_md.read_text(encoding="utf-8")
    hype_count = _term_count(text, HYPE_TERMS)
    scores = _score_from_metrics(report, hype_count=hype_count)
    findings = report["findings"]
    accuracy_rules = {
        "host_stage_directions",
        "missing_host_markers",
        "dropped_transitions",
        "missing_executive_transition_markers",
        "truncated_or_hanging_dialogue",
    }
    listener_rules = {
        "question_density",
        "question_ending_rate",
        "exclamation_density",
        "yes_and_overuse",
        "formulaic_attention_question",
        "formulaic_agreement_opener",
    }
    accuracy_blockers = [
        finding["message"] for finding in findings if finding["rule"] in accuracy_rules
    ]
    listener_blockers = [
        finding["message"] for finding in findings if finding["rule"] in listener_rules
    ]
    if hype_count > 40:
        listener_blockers.append("Hype phrase density is high enough to make the narration feel fluffy.")

    hard_lint_failures = [
        finding["message"]
        for finding in findings
        if finding["severity"] == "fail" and finding["rule"] not in {"conference_label_repetition"}
    ]
    approve_for_tts = (
        not hard_lint_failures
        and not accuracy_blockers
        and not listener_blockers
        and min(scores.values()) >= 3
    )
    return {
        "approve_for_tts": approve_for_tts,
        "script_md": str(script_md),
        "source_script_md": str(source_script_md) if source_script_md else None,
        "deck_kind": deck_kind,
        "topic_id": topic_id,
        "scores": scores,
        "hype_phrase_count": hype_count,
        "accuracy_blockers": accuracy_blockers,
        "listener_blockers": listener_blockers,
        "revision_brief": _revision_brief(report, hype_count=hype_count),
        "lint_status": report["status"],
        "lint_metrics": report["metrics"],
        "lint_findings": findings,
    }


def render_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Narration Review",
        "",
        f"**Script:** `{review['script_md']}`",
        f"**Deck kind:** `{review['deck_kind']}`",
        f"**Approve for TTS:** {'yes' if review['approve_for_tts'] else 'no'}",
        "",
        "## Scores",
        "",
        "| Dimension | Score |",
        "|---|---:|",
    ]
    for key, value in review["scores"].items():
        lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
    lines.extend(["", "## Accuracy Blockers", ""])
    if review["accuracy_blockers"]:
        lines.extend(f"- {item}" for item in review["accuracy_blockers"])
    else:
        lines.append("- None detected by deterministic checks.")
    lines.extend(["", "## Listener Blockers", ""])
    if review["listener_blockers"]:
        lines.extend(f"- {item}" for item in review["listener_blockers"])
    else:
        lines.append("- None detected by deterministic checks.")
    lines.extend(["", "## Revision Brief", ""])
    lines.extend(f"- {item}" for item in review["revision_brief"])
    lines.extend(["", "## Lint Findings", ""])
    if review["lint_findings"]:
        for finding in review["lint_findings"]:
            metric = f" `{finding['metric']}`={finding['value']}" if finding.get("metric") else ""
            lines.append(f"- **{finding['severity'].upper()} {finding['rule']}:** {finding['message']}{metric}")
    else:
        lines.append("- None.")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-md", type=Path, required=True)
    parser.add_argument("--source-script-md", type=Path)
    parser.add_argument("--lint-report", type=Path)
    parser.add_argument(
        "--deck-kind",
        default="topic",
        choices=["topic", "executive", "executive_short", "executive_long"],
    )
    parser.add_argument("--topic-id")
    parser.add_argument("--conference-label")
    parser.add_argument("--review-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    review = review_narration_script(
        script_md=args.script_md,
        source_script_md=args.source_script_md,
        lint_report=args.lint_report,
        deck_kind=args.deck_kind,
        topic_id=args.topic_id,
        conference_label=args.conference_label,
    )
    if args.review_out:
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        args.review_out.write_text(render_markdown(review), encoding="utf-8")
    else:
        print(render_markdown(review), end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"narration review: approve_for_tts={'yes' if review['approve_for_tts'] else 'no'}",
        file=sys.stderr,
    )
    return 0 if review["approve_for_tts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
