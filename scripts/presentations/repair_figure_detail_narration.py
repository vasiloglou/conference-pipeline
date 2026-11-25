#!/usr/bin/env python3
"""Repair figure-detail narration from source-grounded figure specs.

This is a post-generation safety net for presentation narration. It rewrites
only figure-related narration using the current deck asset inventory so a
script cannot keep describing stale bullet-box diagrams after figures have
been regenerated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


HEADER_RE = re.compile(r"(?m)^### \[(?P<kind>SLIDE|ZOOM)\s+(?P<number>\d+)\]\s*(?P<title>.+)$")
SAY_RE = re.compile(r"(?ms)(\*\*SAY:\*\*\s*\n)(.*?)(?=\n\*\*[A-Z][A-Z ]*:?|\n---\s*$|\Z)")
DIALOGUE_RE = re.compile(
    r"(?ms)(\*\*DIALOGUE:\*\*\s*\n)(.*?)(?=\n\*\*TRANSITION MARKER:\*\*|\n---\s*$|\Z)"
)
FIGURE_LINE_RE = re.compile(r"(?m)^\*\*FIGURE:\*\*.*$")
TURN_RE = re.compile(r"(?m)^\s*-\s+\*\*(?P<speaker>[^*:]+):\*\*")


BAD_TEXT_PATTERNS = (
    re.compile(r"\b(?:four|three|two|colored|large|final|numbered)?\s*boxes?\s+(?:labeled|represent|break down|highlight|say)\b", re.I),
    re.compile(r"\b(?:abstract visualization|colored ellipses|overlapping,? semi-transparent ellipses|search landscape)\b", re.I),
    re.compile(r"\b(?:let['’]?s zoom in on this (?:diagram|workflow|chart)|if we look closely at this (?:diagram|chart|timeline))\b", re.I),
    re.compile(r"\b(?:some error metric|proxy for reasoning cost|from 4,000 the|the red line, our ['\"]the|a an |more architectures|limits limits)\b", re.I),
    re.compile(r"\b(?:the|a|an)\s+(?:the|a|an)\b|\b([A-Za-z]{4,})\s+\1\b", re.I),
)


@dataclass
class RepairStats:
    deck_id: str
    presenter_zoom_blocks_repaired: int = 0
    presenter_figure_lines_repaired: int = 0
    notebooklm_blocks_repaired: int = 0
    notebooklm_json_slides_repaired: int = 0
    notebooklm_json_segments_repaired: int = 0
    figure_specs_loaded: int = 0
    warnings: list[str] = field(default_factory=list)


def _clean_phrase(text: object) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = cleaned.replace("→", " to ")
    cleaned = cleaned.replace("—", "-")
    cleaned = cleaned.replace("–", "-")
    cleaned = re.sub(r"\bboxes?\b", "nodes", cleaned, flags=re.I)
    cleaned = re.sub(r"\bellipses\b", "regions", cleaned, flags=re.I)
    cleaned = re.sub(r"\bLatEnt\s+latent\s+state\b", "LatEnt state", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ;,")


def _items(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        cleaned = _clean_phrase(item)
        if cleaned and cleaned.lower() not in {part.lower() for part in out}:
            out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _sentence(text: str) -> str:
    text = _clean_phrase(text)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _lower_initial(text: str) -> str:
    text = _clean_phrase(text)
    if not text:
        return text
    return text[0].lower() + text[1:]


def _spec_slide_number(row: dict[str, Any]) -> int | None:
    for source in (row.get("figure_spec"), row):
        if not isinstance(source, dict):
            continue
        try:
            return int(source.get("slide_number"))
        except (TypeError, ValueError):
            continue
    return None


def load_figure_specs(deck_dir: Path) -> dict[int, dict[str, Any]]:
    candidates = [
        deck_dir / "asset_inventory.json",
        deck_dir / "assets" / "generated" / "figure_specs.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("figure_specs") or payload.get("figures") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        specs: dict[int, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            spec = row.get("figure_spec") if isinstance(row.get("figure_spec"), dict) else row
            if not isinstance(spec, dict):
                continue
            slide_number = _spec_slide_number(row)
            if slide_number is None:
                continue
            specs[slide_number] = spec
        if specs:
            return specs
    return {}


def figure_summary(spec: dict[str, Any]) -> str:
    entities = _items(spec.get("entities"), limit=4)
    relationships = _items(spec.get("relationships"), limit=2)
    quantities = _items(spec.get("quantities"), limit=1)
    semantic_goal = _clean_phrase(spec.get("semantic_goal"))
    depiction = _clean_phrase(spec.get("depiction_type")).replace("_", " ")

    parts: list[str] = []
    if entities:
        parts.append(f"The figure centers on {_join(entities)}")
    elif depiction:
        parts.append(f"The figure uses a {depiction}")
    if relationships:
        parts.append(f"to show {relationships[0]}")
    elif semantic_goal:
        parts.append(f"to show {semantic_goal}")
    summary = "; ".join(parts) if parts else "The figure grounds the slide's visual claim"
    if quantities:
        summary += f", anchored by {quantities[0]}"
    return _sentence(summary)


def presenter_zoom_text(spec: dict[str, Any], *, variant: int) -> str:
    entities = _items(spec.get("entities"), limit=4)
    relationships = _items(spec.get("relationships"), limit=3)
    quantities = _items(spec.get("quantities"), limit=2)
    must_depict = _items(spec.get("must_depict"), limit=2)
    semantic_goal = _clean_phrase(spec.get("semantic_goal"))

    if entities:
        entity_templates = [
            "{items} define the visual anchors",
            "The visual anchors are {items}",
            "{items} are the pieces to keep in view",
            "Start with {items}",
            "{items} give the slide its concrete evidence",
            "The named elements here are {items}",
        ]
        first = _sentence(entity_templates[variant % len(entity_templates)].format(items=_join(entities)))
    else:
        first = _sentence(semantic_goal)

    if relationships:
        relationship_templates = [
            "{rel}; that is the evidence path",
            "{rel}; that is the operational link",
            "{rel}; this is the causal move",
            "{rel}; this is what the slide asks us to notice",
            "{rel}; this is the concrete dependency",
            "{rel}; this is where the claim becomes inspectable",
        ]
        second = _sentence(
            relationship_templates[variant % len(relationship_templates)].format(rel=relationships[0])
        )
    elif must_depict:
        second = _sentence(f"{must_depict[0]}; that is the visible cue")
    else:
        second = _sentence(semantic_goal)

    if quantities:
        if len(quantities) == 1:
            quantity_templates = [
                "{qty} is the number to carry forward",
                "{qty} is the measurement that grounds the visual",
                "{qty} keeps the claim bounded",
                "{qty} is the empirical anchor",
            ]
        else:
            quantity_templates = [
                "{qty} are the reference points to carry forward",
                "{qty} are the measurements that ground the visual",
                "{qty} keep the claim bounded",
                "{qty} are the empirical anchors",
            ]
        third = _sentence(quantity_templates[variant % len(quantity_templates)].format(qty=_join(quantities)))
    elif len(relationships) > 1:
        suffixes = [
            "that is the second constraint",
            "that explains the practical implication",
            "that is why the visual matters",
            "that gives the slide its next consequence",
        ]
        third = _sentence(f"{relationships[1]}; {suffixes[variant % len(suffixes)]}")
    elif semantic_goal:
        third = _sentence(f"Taken together, the visual supports {_lower_initial(semantic_goal)}")
    else:
        third = ""

    return " ".join(part for part in (first, second, third) if part)


def notebook_turns(spec: dict[str, Any], *, host_a: str, host_b: str, variant: int) -> list[dict[str, str]]:
    entities = _items(spec.get("entities"), limit=4)
    relationships = _items(spec.get("relationships"), limit=3)
    quantities = _items(spec.get("quantities"), limit=2)
    semantic_goal = _clean_phrase(spec.get("semantic_goal"))
    depiction = _clean_phrase(spec.get("depiction_type")).replace("_", " ")

    if relationships:
        a_templates = [
            "The useful read is",
            "This visual is really about",
            "The mechanism to follow is",
            "The evidence flow is",
            "The visual dependency is",
            "The important movement is",
        ]
        suffixes = [
            "inside the depicted system",
            "as the central evidence path",
            "rather than as a generic diagram",
            "within the workflow",
            "as the practical dependency",
            "as the mechanism on screen",
        ]
        a_text = _sentence(
            f"{a_templates[variant % len(a_templates)]} {relationships[0]} "
            f"{suffixes[variant % len(suffixes)]}"
        )
    elif entities:
        a_text = _sentence(f"The figure is built around {_join(entities)}")
    elif depiction:
        a_text = _sentence(f"The figure uses a {depiction} to ground the slide")
    else:
        a_text = _sentence(semantic_goal)

    if quantities:
        b_templates = [
            "The anchor is",
            "That number matters:",
            "The constraint to remember is",
            "The evidence is pinned to",
        ]
        b_seed = f"{b_templates[variant % len(b_templates)]} {_join(quantities)}"
    elif len(relationships) > 1:
        b_seed = f"The implication is {relationships[1]}"
    else:
        b_seed = f"The implication is {semantic_goal}"
    b_text = _sentence(b_seed)

    if variant % 3 == 0 and entities:
        a_text = _sentence(
            f"The visual anchors are {_join(entities[:3])}; together they define the mechanism"
        )
    if variant % 4 == 0 and semantic_goal:
        b_text = _sentence(f"That visual framing matters because {_lower_initial(semantic_goal)}")

    return [{"speaker": host_a, "text": a_text}, {"speaker": host_b, "text": b_text}]


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" for line in re.split(r"(?<=\.)\s+", text) if line.strip()) + "\n"


def _replace_say(block: str, text: str) -> tuple[str, bool]:
    repaired, count = SAY_RE.subn(lambda match: match.group(1) + _blockquote(text), block, count=1)
    return repaired, bool(count)


def _replace_dialogue(block: str, turns: list[dict[str, str]]) -> tuple[str, bool]:
    dialogue = "".join(f"- **{turn['speaker']}:** {turn['text']}\n" for turn in turns)
    repaired, count = DIALOGUE_RE.subn(lambda match: match.group(1) + dialogue, block, count=1)
    return repaired, bool(count)


def _replace_figure_line(block: str, spec: dict[str, Any]) -> tuple[str, bool]:
    line = f"**FIGURE:** {figure_summary(spec)}"
    repaired, count = FIGURE_LINE_RE.subn(line, block, count=1)
    return repaired, bool(count)


def _slide_blocks(text: str) -> list[tuple[re.Match[str], str]]:
    headers = list(HEADER_RE.finditer(text))
    blocks: list[tuple[re.Match[str], str]] = []
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        blocks.append((match, text[start:end]))
    return blocks


def _assemble_with_repaired_blocks(text: str, repaired: list[str]) -> str:
    headers = list(HEADER_RE.finditer(text))
    if not headers:
        return text
    pieces = [text[: headers[0].start()]]
    pieces.extend(repaired)
    return "".join(pieces)


def repair_presenter_markdown(path: Path, specs: dict[int, dict[str, Any]], stats: RepairStats) -> None:
    if not path.exists():
        stats.warnings.append(f"missing presenter script: {path}")
        return
    text = path.read_text(encoding="utf-8")
    repaired_blocks: list[str] = []
    changed = False
    for index, (match, block) in enumerate(_slide_blocks(text)):
        number = int(match.group("number"))
        title = match.group("title")
        spec = specs.get(number)
        if spec:
            block, figure_changed = _replace_figure_line(block, spec)
            if figure_changed:
                stats.presenter_figure_lines_repaired += 1
                changed = True
        if match.group("kind") == "ZOOM" and "figure detail" in title.lower() and spec:
            block, say_changed = _replace_say(block, presenter_zoom_text(spec, variant=index))
            block, figure_changed = _replace_figure_line(block, spec)
            if say_changed:
                stats.presenter_zoom_blocks_repaired += 1
                changed = True
            if figure_changed:
                stats.presenter_figure_lines_repaired += 1
                changed = True
        repaired_blocks.append(block)
    if changed:
        path.write_text(_assemble_with_repaired_blocks(text, repaired_blocks), encoding="utf-8")


def _speakers_from_markdown(text: str) -> tuple[str, str]:
    speakers: list[str] = []
    for match in TURN_RE.finditer(text):
        speaker = match.group("speaker").strip()
        if speaker not in speakers:
            speakers.append(speaker)
        if len(speakers) >= 2:
            break
    return (speakers[0], speakers[1]) if len(speakers) >= 2 else ("Iapetus", "Erinome")


def repair_notebook_markdown(path: Path, specs: dict[int, dict[str, Any]], stats: RepairStats) -> None:
    if not path.exists():
        stats.warnings.append(f"missing NotebookLM script: {path}")
        return
    text = path.read_text(encoding="utf-8")
    host_a, host_b = _speakers_from_markdown(text)
    repaired_blocks: list[str] = []
    changed = False
    for index, (match, block) in enumerate(_slide_blocks(text)):
        number = int(match.group("number"))
        source_number = number - 100000 if number >= 100000 else number
        title = match.group("title")
        spec = specs.get(source_number)
        if number >= 100000 and "figure detail" in title.lower() and spec:
            block, block_changed = _replace_dialogue(
                block, notebook_turns(spec, host_a=host_a, host_b=host_b, variant=index)
            )
            if block_changed:
                stats.notebooklm_blocks_repaired += 1
                changed = True
        repaired_blocks.append(block)
    if changed:
        path.write_text(_assemble_with_repaired_blocks(text, repaired_blocks), encoding="utf-8")


def repair_notebook_json(path: Path, specs: dict[int, dict[str, Any]], stats: RepairStats) -> None:
    if not path.exists():
        stats.warnings.append(f"missing NotebookLM script JSON: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    hosts = data.get("hosts") if isinstance(data.get("hosts"), dict) else {}
    host_a = str(hosts.get("host_a") or hosts.get("a") or "Iapetus")
    host_b = str(hosts.get("host_b") or hosts.get("b") or "Erinome")

    turns_by_slide: dict[int, list[dict[str, str]]] = {}
    for index, slide in enumerate(data.get("metadata", {}).get("slides", [])):
        if not isinstance(slide, dict):
            continue
        try:
            number = int(slide.get("slide_number"))
        except (TypeError, ValueError):
            continue
        source_number = number - 100000 if number >= 100000 else number
        spec = specs.get(source_number)
        if number < 100000 or not spec:
            continue
        turns = notebook_turns(spec, host_a=host_a, host_b=host_b, variant=index)
        slide["turns"] = turns
        slide["say"] = " ".join(turn["text"] for turn in turns)
        turns_by_slide[number] = turns
        stats.notebooklm_json_slides_repaired += 1

    new_segments: list[dict[str, Any]] = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        try:
            number = int(segment.get("slide_number"))
        except (TypeError, ValueError):
            number = -1
        turns = turns_by_slide.get(number)
        if not turns:
            new_segments.append(segment)
            continue
        if not any(
            existing.get("slide_number") == number
            and str(existing.get("speaker")) == turns[0]["speaker"]
            and str(existing.get("text")) == turns[0]["text"]
            for existing in new_segments
        ):
            for turn in turns:
                new_segment = dict(segment)
                new_segment["speaker"] = turn["speaker"]
                new_segment["text"] = turn["text"]
                new_segments.append(new_segment)
                stats.notebooklm_json_segments_repaired += 1
    for index, segment in enumerate(new_segments, start=1):
        segment["index"] = index
    if stats.notebooklm_json_slides_repaired or stats.notebooklm_json_segments_repaired:
        data["segments"] = new_segments
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bad_figure_detail_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for match, block in _slide_blocks(path.read_text(encoding="utf-8")):
        number = int(match.group("number"))
        title = match.group("title").lower()
        if number < 100000 and "figure detail" not in title:
            continue
        if any(pattern.search(block) for pattern in BAD_TEXT_PATTERNS):
            count += 1
    return count


def repair_deck(deck_dir: Path) -> dict[str, Any]:
    stats = RepairStats(deck_id=deck_dir.name)
    specs = load_figure_specs(deck_dir)
    stats.figure_specs_loaded = len(specs)
    if not specs:
        stats.warnings.append(f"no figure specs found in {deck_dir}")
    repair_presenter_markdown(deck_dir / "6_presenter_script.md", specs, stats)
    repair_notebook_markdown(deck_dir / "6_presenter_script_notebooklm_2host.md", specs, stats)
    repair_notebook_json(deck_dir / "video_build_notebooklm" / "script.json", specs, stats)
    payload = asdict(stats)
    payload["remaining_presenter_bad_figure_blocks"] = bad_figure_detail_count(
        deck_dir / "6_presenter_script.md"
    )
    payload["remaining_notebooklm_bad_figure_blocks"] = bad_figure_detail_count(
        deck_dir / "6_presenter_script_notebooklm_2host.md"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--deck-dir", type=Path)
    target.add_argument("--decks-dir", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.deck_dir:
        decks = [args.deck_dir]
    else:
        decks = sorted(
            path
            for path in args.decks_dir.iterdir()
            if path.is_dir()
            and not path.name.startswith("_")
            and (path / "asset_inventory.json").exists()
            and (
                (path / "6_presenter_script.md").exists()
                or (path / "6_presenter_script_notebooklm_2host.md").exists()
            )
        )
    reports = [repair_deck(deck) for deck in decks]
    summary = {
        "status": "passed"
        if not any(
            row["remaining_presenter_bad_figure_blocks"] or row["remaining_notebooklm_bad_figure_blocks"]
            for row in reports
        )
        else "warning",
        "deck_count": len(reports),
        "presenter_zoom_blocks_repaired": sum(row["presenter_zoom_blocks_repaired"] for row in reports),
        "presenter_figure_lines_repaired": sum(row["presenter_figure_lines_repaired"] for row in reports),
        "notebooklm_blocks_repaired": sum(row["notebooklm_blocks_repaired"] for row in reports),
        "notebooklm_json_slides_repaired": sum(row["notebooklm_json_slides_repaired"] for row in reports),
        "notebooklm_json_segments_repaired": sum(row["notebooklm_json_segments_repaired"] for row in reports),
        "remaining_presenter_bad_figure_blocks": sum(row["remaining_presenter_bad_figure_blocks"] for row in reports),
        "remaining_notebooklm_bad_figure_blocks": sum(row["remaining_notebooklm_bad_figure_blocks"] for row in reports),
    }
    payload = {"summary": summary, "decks": reports}
    output = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    print(
        "figure narration repair: "
        f"{summary['presenter_zoom_blocks_repaired']} presenter zooms, "
        f"{summary['notebooklm_blocks_repaired']} NotebookLM blocks, "
        f"{summary['remaining_presenter_bad_figure_blocks']} remaining presenter flags, "
        f"{summary['remaining_notebooklm_bad_figure_blocks']} remaining NotebookLM flags",
        file=sys.stderr,
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
