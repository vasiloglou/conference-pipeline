#!/usr/bin/env python3
"""Repair generated two-host narration scripts before TTS.

The script handles deterministic narration fixes and can also apply exact
editorial replacements supplied by a reviewer or LLM revision pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TURN_LINE_RE = re.compile(r"^(?P<prefix>\s*-\s+\*\*(?P<speaker>[^*:]+):\*\*\s*)(?P<text>.+?)(?P<suffix>\s*)$")
SLIDE_HEADER_RE = re.compile(r"(?m)^### \[(?:SLIDE|ZOOM)\s+(?P<number>\d+)\]\s*(?P<title>.+)$")
TRANSITION_MARKER_RE = re.compile(r"(?m)^\*\*TRANSITION MARKER:\*\*\s*(.+?)\s*$")
FORMULAIC_ATTENTION_RE = re.compile(r"\bWhat should (?:we|listeners) pay closest attention to (?:here|on this slide)\?", re.I)
AGREEMENT_OPENER_RE = re.compile(r"^(Exactly|Precisely)\.\s*", re.I)


ATTENTION_REPLACEMENTS = [
    "The practical test is whether that evidence changes the decision a team would make.",
    "The important observation is that the claim now has an operational consequence.",
    "The useful detail is where the evidence changes the threshold for trusting the system.",
    "The takeaway is that this has to alter what researchers measure next.",
    "The hinge is whether the result survives contact with a real workflow.",
]


@dataclass
class RepairStats:
    exact_replacements: int = 0
    attention_questions: int = 0
    agreement_openers: int = 0
    conference_label_mentions_replaced: int = 0
    conference_label_mentions_preserved: int = 0
    slide_separator_repairs: int = 0
    executive_transition_markers_added: int = 0
    markdown_turns_changed: int = 0
    json_turns_changed: int = 0


@dataclass
class ConferenceLabelPolicy:
    label: str
    limit: int
    preserve_every_slides: int
    slide_count: int
    kept: int = 0


def _conference_label_replacement(text_before_match: str) -> str:
    before = text_before_match.lower()
    if before.endswith(("the ", "this ", "that ")):
        return "conference"
    return "the conference"


def _should_preserve_conference_label(
    policy: ConferenceLabelPolicy,
    *,
    slide_index: int | None,
    is_transition: bool,
) -> bool:
    if slide_index in {0, max(0, policy.slide_count - 1)}:
        return True
    if policy.kept >= policy.limit:
        return False
    if (
        slide_index is not None
        and policy.preserve_every_slides > 0
        and slide_index % policy.preserve_every_slides == 0
    ):
        return True
    return bool(is_transition and policy.kept == 0)


def thin_conference_label(
    text: str,
    *,
    policy: ConferenceLabelPolicy | None,
    stats: RepairStats,
    slide_index: int | None,
    is_transition: bool = False,
) -> str:
    if not policy or not policy.label:
        return text

    pattern = re.compile(re.escape(policy.label), re.I)
    pieces: list[str] = []
    position = 0
    changed = False
    for match in pattern.finditer(text):
        pieces.append(text[position : match.start()])
        if _should_preserve_conference_label(policy, slide_index=slide_index, is_transition=is_transition):
            pieces.append(match.group(0))
            policy.kept += 1
            stats.conference_label_mentions_preserved += 1
        else:
            pieces.append(_conference_label_replacement(text[: match.start()]))
            stats.conference_label_mentions_replaced += 1
            changed = True
        position = match.end()
    if not pieces:
        return text
    pieces.append(text[position:])
    repaired = "".join(pieces)
    return repaired if changed else text


def load_replacements(path: Path | None) -> list[tuple[str, str]]:
    if not path:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "replacements" in payload:
        rows = payload["replacements"]
        return [(str(row["old"]), str(row["new"])) for row in rows]
    if isinstance(payload, dict):
        return [(str(old), str(new)) for old, new in payload.items()]
    raise SystemExit(f"Replacement JSON must be an object or contain a replacements list: {path}")


def transition_between(current_title: str, next_title: str) -> str:
    current = current_title.strip().rstrip(".")
    next_ = next_title.strip().rstrip(".")
    if not next_:
        return "The summary moves to the next operating implication."
    if not current:
        return f"The summary now turns to {next_.lower()}."
    return f"That leads into {next_}."


def repair_text(
    text: str,
    *,
    replacements: list[tuple[str, str]],
    stats: RepairStats,
    variant_index: int,
    conference_policy: ConferenceLabelPolicy | None = None,
    slide_index: int | None = None,
    is_transition: bool = False,
) -> str:
    repaired = text
    for old, new in replacements:
        if old in repaired:
            count = repaired.count(old)
            repaired = repaired.replace(old, new)
            stats.exact_replacements += count

    # Provider dialogue occasionally ends a turn with a Unicode ellipsis after
    # cutting a word mid-token.  TTS must never receive that fragment.  Keep the
    # last complete sentence and discard only the incomplete tail.
    if "…" in repaired:
        prefix = repaired.split("…", 1)[0].rstrip()
        boundary = max(prefix.rfind(". "), prefix.rfind("! "), prefix.rfind("? "))
        repaired = (prefix[: boundary + 1] if boundary >= 0 else prefix).rstrip()

    # An unmatched generated quote is a stage-direction artifact, not useful
    # prosody.  Removing quote glyphs preserves the spoken words and makes the
    # deterministic hanging-fragment gate truthful.
    if repaired.count('"') % 2:
        repaired = repaired.replace('"', "")

    if FORMULAIC_ATTENTION_RE.search(repaired):
        replacement = ATTENTION_REPLACEMENTS[variant_index % len(ATTENTION_REPLACEMENTS)]
        repaired = FORMULAIC_ATTENTION_RE.sub(replacement, repaired)
        stats.attention_questions += 1

    opener_match = AGREEMENT_OPENER_RE.match(repaired)
    if opener_match:
        rest = repaired[opener_match.end() :].strip()
        if rest:
            repaired = rest[0].upper() + rest[1:]
        else:
            repaired = "That is the key point."
        stats.agreement_openers += 1

    repaired = thin_conference_label(
        repaired,
        policy=conference_policy,
        stats=stats,
        slide_index=slide_index,
        is_transition=is_transition,
    )

    return repaired


def repair_markdown_text(
    text: str,
    *,
    deck_kind: str,
    replacements: list[tuple[str, str]],
    stats: RepairStats,
    conference_label: str | None = None,
    conference_label_limit: int = 0,
    conference_label_preserve_every_slides: int = 20,
) -> str:
    text, separator_repairs = re.subn(r"(?m)^---(?=### \[(?:SLIDE|ZOOM)\s+\d+\])", "---\n\n", text)
    stats.slide_separator_repairs += separator_repairs
    headers = list(SLIDE_HEADER_RE.finditer(text))
    if not headers:
        return text
    conference_policy = (
        ConferenceLabelPolicy(
            label=conference_label,
            limit=conference_label_limit,
            preserve_every_slides=conference_label_preserve_every_slides,
            slide_count=len(headers),
        )
        if conference_label and conference_label_limit > 0
        else None
    )

    repaired_blocks: list[str] = []
    repaired_blocks.append(text[: headers[0].start()])
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[start:end]
        current_title = match.group("title")
        next_title = headers[index + 1].group("title") if index + 1 < len(headers) else ""

        lines: list[str] = []
        for line_number, line in enumerate(block.splitlines(), start=1):
            line_match = TURN_LINE_RE.match(line)
            if line_match:
                repaired = repair_text(
                    line_match.group("text"),
                    replacements=replacements,
                    stats=stats,
                    variant_index=index + line_number,
                    conference_policy=conference_policy,
                    slide_index=index,
                )
                if repaired != line_match.group("text"):
                    stats.markdown_turns_changed += 1
                line = f"{line_match.group('prefix')}{repaired}{line_match.group('suffix')}"
            elif line.startswith("**TRANSITION MARKER:**"):
                marker_prefix = "**TRANSITION MARKER:**"
                marker_text = line[len(marker_prefix) :].strip()
                repaired = repair_text(
                    marker_text,
                    replacements=replacements,
                    stats=stats,
                    variant_index=index + line_number,
                    conference_policy=conference_policy,
                    slide_index=index,
                    is_transition=True,
                )
                if repaired != marker_text:
                    line = f"{marker_prefix} {repaired}"
            lines.append(line)
        block = "\n".join(lines)
        if not block.endswith("\n"):
            block += "\n"

        if deck_kind.startswith("executive") and index < len(headers) - 1 and not TRANSITION_MARKER_RE.search(block):
            marker = f"**TRANSITION MARKER:** {transition_between(current_title, next_title)}"
            if "\n---" in block:
                block = block.replace("\n---", f"\n\n{marker}\n\n---", 1)
            else:
                block = f"{block.rstrip()}\n\n{marker}\n"
            stats.executive_transition_markers_added += 1
        repaired_blocks.append(block)

    return "".join(repaired_blocks)


def repair_script_json(
    data: dict[str, Any],
    *,
    deck_kind: str,
    replacements: list[tuple[str, str]],
    stats: RepairStats,
    conference_label: str | None = None,
    conference_label_limit: int = 0,
    conference_label_preserve_every_slides: int = 20,
) -> dict[str, Any]:
    # Delivery scripts keep slides under metadata; engaging_narration_raw.json
    # (the TTS source of truth) keeps them at the top level.
    slides = data.get("metadata", {}).get("slides") or data.get("slides", [])
    slide_count = len(slides) if isinstance(slides, list) else 0
    slide_number_to_index: dict[int, int] = {}
    if isinstance(slides, list):
        for index, slide in enumerate(slides):
            try:
                slide_number_to_index[int(slide.get("slide_number", index))] = index
            except (TypeError, ValueError):
                continue
    conference_policy = (
        ConferenceLabelPolicy(
            label=conference_label,
            limit=conference_label_limit,
            preserve_every_slides=conference_label_preserve_every_slides,
            slide_count=slide_count,
        )
        if conference_label and conference_label_limit > 0
        else None
    )
    for segment in data.get("segments", []):
        original = str(segment.get("text") or "")
        try:
            slide_number = int(segment.get("slide_number", segment.get("index", 0)) or 0)
        except (TypeError, ValueError):
            slide_number = 0
        slide_index = slide_number_to_index.get(slide_number, slide_number)
        repaired = repair_text(
            original,
            replacements=replacements,
            stats=stats,
            variant_index=int(segment.get("index", 0)),
            conference_policy=conference_policy,
            slide_index=slide_index,
        )
        if repaired != original:
            segment["text"] = repaired
            stats.json_turns_changed += 1

    if isinstance(slides, list):
        for index, slide in enumerate(slides):
            # Engaging narration keeps its turns under "two_host" (and the
            # single-speaker read under "single"); it is the TTS source of
            # truth, so a fix that skips it never reaches the audio.
            turns = slide.get("turns") or slide.get("two_host") or []
            for turn_index, turn in enumerate(turns):
                original = str(turn.get("text") or "")
                repaired = repair_text(
                    original,
                    replacements=replacements,
                    stats=stats,
                    variant_index=index + turn_index,
                    conference_policy=conference_policy,
                    slide_index=index,
                )
                if repaired != original:
                    turn["text"] = repaired
                    stats.json_turns_changed += 1
            original_single = slide.get("single")
            if isinstance(original_single, str) and original_single:
                repaired_single = repair_text(
                    original_single,
                    replacements=replacements,
                    stats=stats,
                    variant_index=index,
                    conference_policy=conference_policy,
                    slide_index=index,
                )
                if repaired_single != original_single:
                    slide["single"] = repaired_single
                    stats.json_turns_changed += 1
            original_transition = str(slide.get("transition") or "")
            if original_transition:
                repaired_transition = repair_text(
                    original_transition,
                    replacements=replacements,
                    stats=stats,
                    variant_index=index,
                    conference_policy=conference_policy,
                    slide_index=index,
                    is_transition=True,
                )
                if repaired_transition != original_transition:
                    slide["transition"] = repaired_transition
                    stats.json_turns_changed += 1
            if (
                deck_kind.startswith("executive")
                and index < len(slides) - 1
                and not str(slide.get("transition") or "").strip()
            ):
                next_title = str(slides[index + 1].get("title") or "")
                slide["transition"] = transition_between(str(slide.get("title") or ""), next_title)
                stats.executive_transition_markers_added += 1
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-md", type=Path, required=True)
    parser.add_argument("--script-json", type=Path)
    parser.add_argument(
        "--deck-kind",
        default="topic",
        choices=["topic", "executive", "executive_short", "executive_long"],
    )
    parser.add_argument("--replacement-json", type=Path)
    parser.add_argument("--conference-label")
    parser.add_argument("--conference-label-limit", type=int, default=0)
    parser.add_argument("--conference-label-preserve-every-slides", type=int, default=20)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    replacements = load_replacements(args.replacement_json)
    stats = RepairStats()

    original_md = args.script_md.read_text(encoding="utf-8")
    repaired_md = repair_markdown_text(
        original_md,
        deck_kind=args.deck_kind,
        replacements=replacements,
        stats=stats,
        conference_label=args.conference_label,
        conference_label_limit=args.conference_label_limit,
        conference_label_preserve_every_slides=args.conference_label_preserve_every_slides,
    )
    if not args.dry_run and repaired_md != original_md:
        args.script_md.write_text(repaired_md, encoding="utf-8")

    if args.script_json:
        data = json.loads(args.script_json.read_text(encoding="utf-8"))
        repaired_data = repair_script_json(
            data,
            deck_kind=args.deck_kind,
            replacements=replacements,
            stats=stats,
            conference_label=args.conference_label,
            conference_label_limit=args.conference_label_limit,
            conference_label_preserve_every_slides=args.conference_label_preserve_every_slides,
        )
        if not args.dry_run:
            args.script_json.write_text(
                json.dumps(repaired_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    report = {
        "script_md": str(args.script_md),
        "script_json": str(args.script_json) if args.script_json else None,
        "deck_kind": args.deck_kind,
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
        "narration fix: "
        f"{stats.markdown_turns_changed} markdown turns, "
        f"{stats.json_turns_changed} json turns, "
        f"{stats.executive_transition_markers_added} transition markers",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
