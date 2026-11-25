#!/usr/bin/env python3
"""Editorial review gate for single-presenter narration before TTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lint_presenter_narration import analyze_presenter_narration  # noqa: E402


def _load_lint_report(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scores(report: dict[str, Any]) -> dict[str, int]:
    rules = {finding["rule"] for finding in report["findings"]}
    fail_count = report["finding_counts"].get("fail", 0)
    warn_count = report["finding_counts"].get("warn", 0)
    scores = {
        "factual_fidelity": 5,
        "cohesion": 5,
        "listener_interest": 5,
        "fluff_control": 5,
        "transition_quality": 5,
        "naturalness": 5,
        "pleasantness": 5,
    }
    if fail_count:
        scores["factual_fidelity"] = 1
        scores["pleasantness"] = 2
    if "dropped_presenter_transitions" in rules:
        scores["cohesion"] = 2
        scores["transition_quality"] = 1
    elif "presenter_transition_preservation_ratio" in rules:
        scores["cohesion"] = 3
        scores["transition_quality"] = 3
    if "welcome_formula_opening" in rules or "closing_formula_opening" in rules:
        scores["naturalness"] = min(scores["naturalness"], 3)
        scores["pleasantness"] = min(scores["pleasantness"], 4)
    if "stitch_template_repetition" in rules or "imperative_glue_template" in rules:
        scores["cohesion"] = min(scores["cohesion"], 2)
        scores["naturalness"] = min(scores["naturalness"], 2)
        scores["pleasantness"] = min(scores["pleasantness"], 2)
    if "closing_grammar_glue" in rules:
        scores["naturalness"] = min(scores["naturalness"], 2)
        scores["pleasantness"] = min(scores["pleasantness"], 2)
    if any(rule.startswith("filler_") for rule in rules):
        scores["fluff_control"] = 3 if warn_count else scores["fluff_control"]
        if fail_count:
            scores["fluff_control"] = 2
    return scores


def _revision_brief(report: dict[str, Any]) -> list[str]:
    rules = {finding["rule"] for finding in report["findings"]}
    brief: list[str] = []
    if "presenter_script_parse_failed" in rules or "missing_presenter_say_blocks" in rules:
        brief.append("Regenerate the presenter markdown as real SAY blocks before TTS.")
    if any(rule.startswith("tts_forbidden_") for rule in rules):
        brief.append("Regenerate script.json from SAY plus TRANSITION only; remove context, refs, arXiv markers, and stage directions from TTS text.")
    if "dropped_presenter_transitions" in rules or "presenter_transition_preservation_ratio" in rules:
        brief.append("Preserve source transitions in script.json so adjacent slides feel connected.")
    if "welcome_formula_opening" in rules:
        brief.append("Rewrite the opening to start with the stake or tension, not a generic greeting.")
    if "closing_formula_opening" in rules:
        brief.append("Vary the closing opener and land on a concrete implication or priority.")
    if "stitch_template_repetition" in rules or "imperative_glue_template" in rules:
        brief.append("Rewrite executive SAY blocks as authored narration from synthesis prose; avoid repeated stitch phrases and imperative bullet glue.")
    if "closing_grammar_glue" in rules:
        brief.append('Repair closing sentence joins such as "The next step is We need..." into clean infinitive or noun-phrase grammar.')
    if any(rule.startswith("filler_") for rule in rules):
        brief.append("Thin repeated filler words such as critical, crucial, key, and fundamental.")
    if "conference_label_repetition" in rules:
        brief.append("Mention the conference label only where it orients the listener.")
    if not brief:
        brief.append("No blocking rewrite needed; keep review focused on factual spot checks and pacing.")
    return brief


def review_presenter_narration(
    *,
    script_md: Path,
    script_json: Path | None = None,
    lint_report: Path | None = None,
    deck_kind: str = "topic",
    topic_id: str | None = None,
    conference_label: str | None = None,
) -> dict[str, Any]:
    report = _load_lint_report(lint_report) or analyze_presenter_narration(
        script_md=script_md,
        script_json=script_json,
        deck_kind=deck_kind,
        topic_id=topic_id,
        conference_label=conference_label,
    )
    scores = _scores(report)
    hard_failures = [finding["message"] for finding in report["findings"] if finding["severity"] == "fail"]
    listener_warnings = [finding["message"] for finding in report["findings"] if finding["severity"] == "warn"]
    approve_for_tts = not hard_failures and min(scores.values()) >= 3
    return {
        "approve_for_tts": approve_for_tts,
        "script_md": str(script_md),
        "script_json": str(script_json) if script_json else None,
        "deck_kind": deck_kind,
        "topic_id": topic_id,
        "scores": scores,
        "accuracy_blockers": hard_failures,
        "listener_warnings": listener_warnings,
        "revision_brief": _revision_brief(report),
        "lint_status": report["status"],
        "lint_metrics": report["metrics"],
        "lint_findings": report["findings"],
    }


def render_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Presenter Narration Review",
        "",
        f"**Script:** `{review['script_md']}`",
        f"**Script JSON:** `{review['script_json']}`",
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
    lines.extend(f"- {item}" for item in review["accuracy_blockers"] or ["None detected by deterministic checks."])
    lines.extend(["", "## Listener Warnings", ""])
    lines.extend(f"- {item}" for item in review["listener_warnings"] or ["None detected by deterministic checks."])
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
    parser.add_argument("--script-json", type=Path)
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
    review = review_presenter_narration(
        script_md=args.script_md,
        script_json=args.script_json,
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
        f"presenter narration review: approve_for_tts={'yes' if review['approve_for_tts'] else 'no'}",
        file=sys.stderr,
    )
    return 0 if review["approve_for_tts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
