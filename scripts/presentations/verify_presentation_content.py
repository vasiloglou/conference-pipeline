#!/usr/bin/env python3
"""Content-richness verifiers for a built topic deck (v2.3 Phase 05).

Deterministic gates calibrated against the approved Topic 4 canary, so a deck
built from a thin skeleton cannot pass silently:

- coverage: distinct sources / named people / distinct works in deck mentions
- duration: narration minutes on both delivery contracts (pre-TTS)
- figure richness: per-figure and deck-average shape/label structure
- template leak: hardcoded topic strings and skeleton boilerplate

Writes `content_verification.json` (+ .md) into the deck dir; exits non-zero
when any gate fails.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/presentations/verify_presentation_content.py \
      --deck-dir conferences/iclr2026/decks/02_agentic_systems
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Canary reference (Topic 4): 33 sources, 146 people, 31 works, 19.8 min,
# figures avg 23 shapes / 26 text elements. Floors sit below the canary but
# far above the skeleton output (5-9 sources, 0-2 people, 9 min, 8-9 shapes).
DEFAULT_MIN_SOURCES = 25
DEFAULT_MIN_PEOPLE = 40
DEFAULT_MIN_WORKS = 20
DEFAULT_MIN_MINUTES = 20.0
DEFAULT_MAX_MINUTES = 32.0
DEFAULT_MIN_FIGURE_SHAPES_AVG = 14.0
DEFAULT_MIN_FIGURE_TEXTS_AVG = 18.0
DEFAULT_MIN_FIGURE_SHAPES_EACH = 8
DEFAULT_MIN_SLIDES = 22

SKELETON_BOILERPLATE = (
    "How the research clusters connect",
    "From source ledger to rendered deck",
    "What the canary deck must prove",
    "the named evidence and what it changes",
    "Training implication: teams building",
    "Evaluation implication: benchmarks that ignore",
    "Deployment implication: because of",
    "Operations implication:",
    "Design implication: because",
)
PLACEHOLDER_PEOPLE = {"unknown", "n/a", "na", "none", "anonymous", "unnamed", "unidentified"}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _gate(gate_id: str, passed: bool, value: Any, threshold: Any, detail: str = "") -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": "pass" if passed else "fail",
        "value": value,
        "threshold": threshold,
        "detail": detail,
    }


def check_coverage(deck: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    sources: set[str] = set()
    people: set[str] = set()
    works: set[str] = set()
    for slide in deck.get("slides") or []:
        for mention in slide.get("named_paper_talk_speaker_author_mentions") or []:
            if not isinstance(mention, dict):
                continue
            if mention.get("source_id"):
                sources.add(str(mention["source_id"]))
            title = str(mention.get("title") or "").strip()
            if title:
                works.add(title.lower()[:70])
            for person in (mention.get("speakers") or []) + (mention.get("authors") or []):
                name = str(person).strip()
                if name and name.lower() not in PLACEHOLDER_PEOPLE:
                    people.add(name)
    slides = deck.get("slides") or []
    return [
        _gate("coverage_sources", len(sources) >= args.min_sources, len(sources), args.min_sources,
              "distinct source_ids mentioned across slides"),
        _gate("coverage_people", len(people) >= args.min_people, len(people), args.min_people,
              "distinct named authors/speakers mentioned across slides"),
        _gate("coverage_works", len(works) >= args.min_works, len(works), args.min_works,
              "distinct work titles mentioned across slides"),
        _gate("coverage_slides", len(slides) >= args.min_slides, len(slides), args.min_slides,
              "deck slide count"),
    ]


def check_duration(deck_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for name in ("delivery_single_speaker.json", "delivery_notebooklm.json"):
        path = deck_dir / name
        if not path.is_file():
            gates.append(_gate(f"duration_{name}", False, None, f"{args.min_minutes}-{args.max_minutes} min", "missing delivery contract"))
            continue
        delivery = _load(path)
        minutes = float(delivery.get("total_duration_seconds") or 0.0) / 60.0
        gates.append(
            _gate(
                f"duration_{name.removesuffix('.json')}",
                args.min_minutes <= minutes <= args.max_minutes,
                round(minutes, 1),
                f"{args.min_minutes}-{args.max_minutes} min",
                "narration duration from the delivery contract (pre-TTS)",
            )
        )
    return gates


def check_figures(deck_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    svgs = sorted((deck_dir / "assets" / "generated" / "phase16").glob("*.svg"))
    if not svgs:
        return [_gate("figure_richness", False, 0, "figures present", "no phase16 figures found")]
    shapes_counts, text_counts, thin = [], [], []
    for svg in svgs:
        text = svg.read_text(encoding="utf-8", errors="replace")
        shapes = len(re.findall(r"<(rect|circle|ellipse|polygon|path|line)\b", text))
        texts = len(re.findall(r"<text\b", text))
        shapes_counts.append(shapes)
        text_counts.append(texts)
        if shapes < args.min_figure_shapes_each:
            thin.append(f"{svg.name[:40]}({shapes} shapes)")
    shapes_avg = sum(shapes_counts) / len(shapes_counts)
    texts_avg = sum(text_counts) / len(text_counts)
    return [
        _gate("figure_shapes_avg", shapes_avg >= args.min_figure_shapes_avg,
              round(shapes_avg, 1), args.min_figure_shapes_avg,
              "average drawn shapes per figure (canary ~23)"),
        _gate("figure_texts_avg", texts_avg >= args.min_figure_texts_avg,
              round(texts_avg, 1), args.min_figure_texts_avg,
              "average text elements per figure (canary ~26)"),
        _gate("figure_shapes_each", not thin, thin or "all ok", f">={args.min_figure_shapes_each} shapes each",
              "figures below the per-figure structure floor"),
    ]


def check_template_leaks(deck_dir: Path, deck: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    deck_id = str(deck.get("presentation_id") or deck_dir.name)
    topic_match = re.match(r"^0*(\d+)_", deck_id)
    html_path = deck_dir / "presentation.html"
    html_text = html_path.read_text(encoding="utf-8", errors="replace") if html_path.is_file() else ""
    wrong_topic: list[str] = []
    if topic_match and html_text:
        own = int(topic_match.group(1))
        for other in set(re.findall(r"Topic (\d+)", html_text)):
            if int(other) != own:
                wrong_topic.append(f"Topic {other}")
    gates.append(_gate("template_topic_string", not wrong_topic, wrong_topic or "none",
                       "no foreign topic numbers in rendered HTML",
                       "hardcoded topic strings leaked from another deck's template"))
    blob = json.dumps(deck) + html_text
    leaks = sorted({marker for marker in SKELETON_BOILERPLATE if marker in blob})
    gates.append(_gate("template_skeleton_boilerplate", not leaks, leaks or "none",
                       "no skeleton-generator boilerplate in deck/HTML",
                       "phrases emitted by the deterministic source-ledger skeleton"))
    return gates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", required=True, type=Path)
    parser.add_argument("--min-sources", type=int, default=DEFAULT_MIN_SOURCES)
    parser.add_argument("--min-people", type=int, default=DEFAULT_MIN_PEOPLE)
    parser.add_argument("--min-works", type=int, default=DEFAULT_MIN_WORKS)
    parser.add_argument("--min-slides", type=int, default=DEFAULT_MIN_SLIDES)
    parser.add_argument("--min-minutes", type=float, default=DEFAULT_MIN_MINUTES)
    parser.add_argument("--max-minutes", type=float, default=DEFAULT_MAX_MINUTES)
    parser.add_argument("--min-figure-shapes-avg", type=float, default=DEFAULT_MIN_FIGURE_SHAPES_AVG)
    parser.add_argument("--min-figure-texts-avg", type=float, default=DEFAULT_MIN_FIGURE_TEXTS_AVG)
    parser.add_argument("--min-figure-shapes-each", type=int, default=DEFAULT_MIN_FIGURE_SHAPES_EACH)
    parser.add_argument("--skip", action="append", default=[],
                        help="Gate id prefix to skip (e.g. 'duration' before narration exists).")
    parser.add_argument("--report-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deck_dir = args.deck_dir.expanduser().resolve()
    deck = _load(deck_dir / "presentation_deck.json")
    gates: list[dict[str, Any]] = []
    gates.extend(check_coverage(deck, args))
    gates.extend(check_duration(deck_dir, args))
    gates.extend(check_figures(deck_dir, args))
    gates.extend(check_template_leaks(deck_dir, deck))
    if args.skip:
        gates = [
            g for g in gates
            if not any(g["gate_id"].startswith(prefix) for prefix in args.skip)
        ]
    failed = [g for g in gates if g["status"] != "pass"]
    payload = {
        "schema_version": 1,
        "verifier": "scripts/presentations/verify_presentation_content.py",
        "deck_dir": str(deck_dir),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "fail" if failed else "pass",
        "gates": gates,
        "calibration_reference": "04_safety_alignment_privacy (approved v2.2 canary)",
    }
    report_path = args.report_out or deck_dir / "content_verification.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# Content verification — {deck.get('presentation_id') or deck_dir.name}", ""]
    for gate in gates:
        mark = "PASS" if gate["status"] == "pass" else "FAIL"
        lines.append(f"- **{mark}** `{gate['gate_id']}`: {gate['value']} (threshold {gate['threshold']}) — {gate['detail']}")
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failed_gates": [g["gate_id"] for g in failed]}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
