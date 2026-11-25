#!/usr/bin/env python3
"""Run the Phase 13 editorial gate across a deck series.

This is the repeatable cross-deck preflight used before bulk TTS/video
regeneration. It wraps presentation_editorial_gate.py for every topic and
executive-summary deck, emits a machine-readable JSON report, and can also emit
a compact Markdown dashboard for operator review.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from presentation_editorial_gate import analyze_deck  # noqa: E402

EXECUTIVE_DECK_KINDS = {
    "executive_summary": "executive",
    "executive_summary_short": "executive_short",
    "executive_summary_long": "executive_long",
}


@dataclass(frozen=True)
class DeckTarget:
    deck_id: str
    deck_dir: Path
    deck_kind: str


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_topic_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) >= 3 and path.name[:2].isdigit() and path.name[2] == "_"


def discover_decks(decks_dir: Path) -> list[DeckTarget]:
    targets: list[DeckTarget] = []
    for deck_dir in sorted(path for path in decks_dir.iterdir() if path.is_dir()):
        if _is_topic_dir(deck_dir):
            targets.append(
                DeckTarget(deck_id=deck_dir.name, deck_dir=deck_dir, deck_kind="topic")
            )
            continue
        deck_kind = EXECUTIVE_DECK_KINDS.get(deck_dir.name)
        if deck_kind:
            targets.append(
                DeckTarget(deck_id=deck_dir.name, deck_dir=deck_dir, deck_kind=deck_kind)
            )
    return targets


def _count_findings(findings: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(finding.get("severity", "unknown")) for finding in findings))


def build_preflight_report(decks_dir: Path) -> dict[str, Any]:
    decks_dir = decks_dir.resolve()
    targets = discover_decks(decks_dir)
    deck_reports: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()

    for target in targets:
        report = analyze_deck(deck_dir=target.deck_dir, deck_kind=target.deck_kind)
        findings = report.get("findings", [])
        if not isinstance(findings, list):
            findings = []
        finding_counts = _count_findings(findings)
        status = str(report.get("status", "unknown"))
        status_counts[status] += 1
        severity_counts.update(finding_counts)
        for finding in findings:
            if finding.get("severity") == "fail":
                rule_counts[str(finding.get("rule", "unknown"))] += 1
        deck_reports.append(
            {
                "deck_id": target.deck_id,
                "deck_kind": target.deck_kind,
                "deck_dir": str(target.deck_dir),
                "status": status,
                "finding_counts": finding_counts,
                "metrics": report.get("metrics", {}),
                "findings": findings,
            }
        )

    failed_decks = [deck["deck_id"] for deck in deck_reports if deck["status"] == "failed"]
    warning_decks = [deck["deck_id"] for deck in deck_reports if deck["status"] == "warning"]
    status = "failed" if failed_decks else "warning" if warning_decks else "passed"
    return {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/run_phase13_editorial_preflight.py",
        "decks_dir": str(decks_dir),
        "status": status,
        "summary": {
            "deck_count": len(deck_reports),
            "failed_deck_count": len(failed_decks),
            "warning_deck_count": len(warning_decks),
            "passed_deck_count": sum(1 for deck in deck_reports if deck["status"] == "passed"),
            "status_counts": dict(status_counts),
            "severity_counts": dict(severity_counts),
            "hard_failure_rule_counts": dict(sorted(rule_counts.items())),
            "failed_decks": failed_decks,
            "warning_decks": warning_decks,
        },
        "decks": deck_reports,
    }


def _duration(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return f"{float(value):.2f}"
    return ""


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Phase 13 Editorial Preflight",
        "",
        f"Generated: `{report.get('generated_at', '')}`",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Decks: `{summary.get('deck_count', 0)}`",
        f"Failed decks: `{summary.get('failed_deck_count', 0)}`",
        f"Warning decks: `{summary.get('warning_deck_count', 0)}`",
        "",
        "## Deck Summary",
        "",
        "| Deck | Kind | Status | Fail | Warn | Slides | Presenter s | NotebookLM s | Zooms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for deck in report.get("decks", []):
        metrics = deck.get("metrics", {}) if isinstance(deck.get("metrics"), dict) else {}
        counts = deck.get("finding_counts", {}) if isinstance(deck.get("finding_counts"), dict) else {}
        lines.append(
            "| {deck_id} | {kind} | {status} | {fail} | {warn} | {slides} | {presenter} | {notebook} | {zooms} |".format(
                deck_id=deck.get("deck_id", ""),
                kind=deck.get("deck_kind", ""),
                status=deck.get("status", ""),
                fail=counts.get("fail", 0),
                warn=counts.get("warn", 0),
                slides=metrics.get("slide_count", ""),
                presenter=_duration(metrics, "presenter_estimated_duration_seconds"),
                notebook=_duration(metrics, "notebooklm_estimated_duration_seconds"),
                zooms=metrics.get("zoom_count", ""),
            )
        )

    rule_counts = summary.get("hard_failure_rule_counts", {})
    if isinstance(rule_counts, dict) and rule_counts:
        lines.extend(["", "## Hard Failure Rules", ""])
        for rule, count in rule_counts.items():
            lines.append(f"- `{rule}`: `{count}`")

    failed_decks = summary.get("failed_decks", [])
    if isinstance(failed_decks, list) and failed_decks:
        lines.extend(["", "## Failed Decks", ""])
        for deck_id in failed_decks:
            deck = next(
                (candidate for candidate in report.get("decks", []) if candidate.get("deck_id") == deck_id),
                None,
            )
            if not isinstance(deck, dict):
                continue
            rules = [
                str(finding.get("rule", "unknown"))
                for finding in deck.get("findings", [])
                if finding.get("severity") == "fail"
            ]
            lines.append(f"- `{deck_id}`: {', '.join(f'`{rule}`' for rule in rules)}")

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decks-dir",
        type=Path,
        required=True,
        help="Directory containing topic/executive deck directories.",
    )
    parser.add_argument("--report-out", type=Path, help="JSON report output path.")
    parser.add_argument("--markdown-out", type=Path, help="Markdown dashboard output path.")
    parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Exit nonzero when any deck has a hard editorial failure.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_preflight_report(args.decks_dir)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    summary = report["summary"]
    print(
        "phase13 editorial preflight: "
        f"{report['status']} "
        f"({summary['failed_deck_count']} failed, {summary['warning_deck_count']} warning, "
        f"{summary['passed_deck_count']} passed)",
        file=sys.stderr,
    )
    return 1 if args.fail_on_fail and report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
