#!/usr/bin/env python3
"""Repair cross-deck repetitive phrasing in generated narration scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SLIDE_HEADER_RE = re.compile(r"(?m)^### \[(?:SLIDE|ZOOM)\s+(?P<number>\d+)\]\s*(?P<title>.+)$")
TURN_LINE_RE = re.compile(r"^(?P<prefix>\s*-\s+\*\*(?P<speaker>[^*:]+):\*\*\s*)(?P<text>.+?)(?P<suffix>\s*)$")
SAY_BLOCK_RE = re.compile(
    r"(?ms)(?P<label>^\*\*SAY:\*\*\s*)(?P<body>.*?)(?=^\*\*[A-Z][A-Z ]*:?\*\*|^---\s*$|\Z)"
)


@dataclass(frozen=True)
class PhraseFamily:
    label: str
    pattern: re.Pattern[str]
    variants: tuple[str, ...]


@dataclass
class RepairStats:
    scripts_seen: int = 0
    markdown_turns_changed: int = 0
    json_turns_changed: int = 0
    replacements_by_family: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.replacements_by_family is None:
            self.replacements_by_family = {}


PHRASE_FAMILIES = (
    PhraseFamily(
        label="design_target",
        pattern=re.compile(r"\bthe design target\b", re.I),
        variants=(
            "the success criterion",
            "the operating goal",
            "the desired behavior",
            "the engineering constraint",
            "the practical threshold",
            "the criterion that matters",
            "the intended behavior",
            "the target outcome",
            "the decision threshold",
            "the design constraint",
            "the performance goal",
            "the operating requirement",
            "the useful signal",
            "the result teams need",
            "the behavior the system has to preserve",
            "the point of the method",
        ),
    ),
    PhraseFamily(
        label="scaling_pressure_curve",
        pattern=re.compile(r"\b(?:the|this|that)?\s*scaling pressure curve\b", re.I),
        variants=(
            "the resource tradeoff",
            "the cost profile",
            "the context-efficiency tradeoff",
            "the long-context bottleneck",
            "the compute tradeoff",
            "the efficiency frontier",
            "the resource slope",
            "the operating frontier",
            "the context-length pressure",
            "the cost-performance pattern",
            "the resource boundary",
            "the memory and latency tradeoff",
            "the context scaling constraint",
            "the efficiency constraint",
            "the cost curve",
            "the systems pressure",
        ),
    ),
    PhraseFamily(
        label="scaling_pressure",
        pattern=re.compile(r"\b(?:the\s+)?scaling pressure\b", re.I),
        variants=(
            "the resource pressure",
            "the systems burden",
            "the cost pressure",
            "the efficiency strain",
            "the serving pressure",
            "the scaling bottleneck",
            "the compute burden",
            "the long-context strain",
        ),
    ),
    PhraseFamily(
        label="flatten_curve",
        pattern=re.compile(r"\bflatten (?:that|this|the) curve\b", re.I),
        variants=(
            "reduce that slope",
            "ease that pressure",
            "lower the cost slope",
            "make the tradeoff less punishing",
            "bend the cost line down",
            "bring the resource cost down",
            "make the scaling behavior gentler",
            "reduce the long-context penalty",
            "soften the efficiency cliff",
            "lower the systems burden",
            "improve that operating frontier",
            "make the resource profile manageable",
            "turn that pressure into a tractable constraint",
            "reduce the cost of longer context",
            "make the bottleneck less severe",
            "improve the context-efficiency tradeoff",
        ),
    ),
    PhraseFamily(
        label="from_4k_to_128k",
        pattern=re.compile(r"\bfrom 4k to 128k(?: tokens)?\b", re.I),
        variants=(
            "across a 32x context jump",
            "from 4,000 to 128,000 tokens",
            "over the long-context range",
            "from short to very long context windows",
            "across the 4,000-to-128,000-token span",
            "through a six-figure-token expansion",
            "as the context budget grows by 32x",
            "across increasingly long contexts",
            "from modest to very large context budgets",
            "through the long-context regime",
            "as the window moves into 128,000-token territory",
            "across a short-to-long context sweep",
        ),
    ),
    PhraseFamily(
        label="4k_to_128k",
        pattern=re.compile(r"\b4k to 128k(?: tokens)?\b", re.I),
        variants=(
            "a 32x context jump",
            "the 4,000-to-128,000-token range",
            "long-context expansion",
            "the short-to-long context span",
            "a move into six-figure token windows",
            "the extended-context range",
            "a 4,000-to-128,000-token sweep",
            "the long-context regime",
            "a large context-budget expansion",
            "the six-figure-token span",
            "the expanded-window setting",
            "the 128,000-token end of the sweep",
        ),
    ),
    PhraseFamily(
        label="to_128k_tokens",
        pattern=re.compile(r"\bto 128k tokens\b", re.I),
        variants=(
            "to 128,000 tokens",
            "into six-figure token windows",
            "toward very long contexts",
            "into the 128,000-token regime",
            "toward the extended-context setting",
            "into much larger context budgets",
            "to a six-figure-token window",
            "toward the long-context end of the sweep",
        ),
    ),
    PhraseFamily(
        label="six_figure_tokens",
        pattern=re.compile(r"\bsix[- ]figure[- ]token(?:s)?(?:\s+(?:window|windows|expansion|span|range|regime))?\b", re.I),
        variants=(
            "very long context",
            "an extended prompt window",
            "the high-context regime",
            "a much larger input span",
            "the extended-token setting",
            "a large context range",
            "the far end of the context sweep",
            "an expanded context window",
        ),
    ),
    PhraseFamily(
        label="size_from_4k",
        pattern=re.compile(
            r"\b(?:the\s+)?(?:(?:available|input|token)\s+)?(?:context|window|model)\s+size,\s*from\s+4k\b",
            re.I,
        ),
        variants=(
            "the context span, beginning at 4K",
            "the input range, starting at 4K",
            "the token-window range, beginning at 4K",
            "the scale proxy, starting at 4K",
            "the prompt span, beginning at 4K",
            "the long-context sweep, starting at 4K",
            "the context range, beginning at 4K",
            "the window range, starting at 4K",
        ),
    ),
    PhraseFamily(
        label="memory_aware_architecture",
        pattern=re.compile(r"\b(?:new\s+)?memory[- ]aware architecture(?:s)?\b", re.I),
        variants=(
            "architectures designed around memory budgets",
            "systems that budget memory explicitly",
            "models built for memory limits",
            "memory-conscious systems",
            "architectures that treat memory as a constraint",
            "system designs with explicit memory budgets",
            "long-context designs that manage memory directly",
            "architectures built around the memory bottleneck",
            "systems engineered for memory pressure",
            "models that make memory use part of the design",
            "architectures with memory constraints in the loop",
            "systems that plan around the cache budget",
            "designs that put memory cost in the foreground",
            "architectures shaped by memory limits",
            "models tuned for the memory footprint",
            "systems that handle long-context memory deliberately",
        ),
    ),
    PhraseFamily(
        label="latency_memory_and",
        pattern=re.compile(r"\blatency,\s*memory,\s*and\b", re.I),
        variants=(
            "runtime cost, memory use, and",
            "speed, footprint, and",
            "response time, cache use, and",
            "serving latency, memory footprint, and",
            "runtime delay, memory demand, and",
            "inference cost, memory pressure, and",
            "speed limits, memory use, and",
            "serving cost, cache pressure, and",
        ),
    ),
    PhraseFamily(
        label="latency_and_memory",
        pattern=re.compile(r"\blatency and memory\b", re.I),
        variants=(
            "latency plus memory use",
            "runtime and memory footprint",
            "systems cost and memory demand",
            "compute delay and memory pressure",
            "serving cost and memory footprint",
            "response time and cache pressure",
            "speed and memory footprint",
            "runtime delay and memory demand",
        ),
    ),
    PhraseFamily(
        label="kv_cache",
        pattern=re.compile(r"\b(?:the\s+)?k[/-]?v[- ]cache\b", re.I),
        variants=(
            "the key-value cache",
            "the attention cache",
            "the cached keys and values",
            "the cache state",
            "the transformer cache",
            "the stored attention state",
            "the inference cache",
            "the long-context cache",
        ),
    ),
    PhraseFamily(
        label="context_window",
        pattern=re.compile(r"\bthe context window\b|\bcontext window\b", re.I),
        variants=(
            "the available context",
            "the prompt window",
            "the token window",
            "the context budget",
            "the long-context window",
            "the usable context",
            "the input window",
            "the context span",
        ),
    ),
    PhraseFamily(
        label="context_length",
        pattern=re.compile(r"\b(?:the\s+)?context length\b", re.I),
        variants=(
            "the prompt span",
            "the available token range",
            "the input span",
            "the usable window",
            "the context allocation",
            "the extended-input span",
            "the token-window size",
            "the long-window budget",
        ),
    ),
    PhraseFamily(
        label="scaffold_value",
        pattern=re.compile(r"\bscaffold value\b", re.I),
        variants=(
            "supporting signal",
            "workflow support",
            "assistant contribution",
            "structured help",
            "scaffolding benefit",
            "support score",
            "guidance effect",
            "task support",
        ),
    ),
    PhraseFamily(
        label="target_outcome",
        pattern=re.compile(r"\b(?:our|the)\s+target outcome\b", re.I),
        variants=(
            "the operating goal",
            "the result teams need",
            "the practical threshold",
            "the desired behavior",
            "the decision criterion",
            "the useful endpoint",
            "the behavior we want",
            "the production criterion",
        ),
    ),
    PhraseFamily(
        label="architecture_you_can",
        pattern=re.compile(
            r"\b(?:(?:in|looking at|zooming in on|if we zoom in on|let(?:'|’)s zoom in on)\s+(?:this\s+|the\s+)?architecture"
            r"|(?:this\s+|the\s+)?architecture),?\s*\.?\s*you can(?: see)?\b",
            re.I,
        ),
        variants=(
            "the diagram shows",
            "the system layout makes clear",
            "the model structure highlights",
            "the pipeline view shows",
            "the figure lays out",
            "the architecture view highlights",
            "the method diagram shows",
            "the design sketch makes visible",
        ),
    ),
    PhraseFamily(
        label="memory_footprint",
        pattern=re.compile(r"\b(?:the\s+)?memory footprint\b", re.I),
        variants=(
            "the memory demand",
            "cache pressure",
            "the memory load",
            "the serving footprint",
            "the memory cost",
            "the cache burden",
            "the runtime footprint",
            "the storage pressure",
        ),
    ),
    PhraseFamily(
        label="shaped_by_memory",
        pattern=re.compile(r"\bshaped by memory\b", re.I),
        variants=(
            "constrained by cache cost",
            "organized around memory limits",
            "bounded by memory pressure",
            "designed around cache budgets",
            "conditioned by memory demand",
            "driven by storage pressure",
            "planned around memory cost",
            "limited by serving footprint",
        ),
    ),
    PhraseFamily(
        label="cost_curve",
        pattern=re.compile(r"\b(?:the\s+)?cost curve\b", re.I),
        variants=(
            "the cost profile",
            "the resource slope",
            "the operating cost",
            "the efficiency profile",
            "the serving-cost pattern",
            "the resource tradeoff",
            "the cost trajectory",
            "the compute profile",
        ),
    ),
    PhraseFamily(
        label="design_target_bare",
        pattern=re.compile(r"\bdesign target\b", re.I),
        variants=(
            "success criterion",
            "operating goal",
            "desired behavior",
            "engineering constraint",
            "practical threshold",
            "decision criterion",
            "target outcome",
            "operating requirement",
        ),
    ),
    PhraseFamily(
        label="curve_is_not",
        pattern=re.compile(r"\b(?:the|this|that)?\s*curve isn[’']?t\b", re.I),
        variants=(
            "the tradeoff is not",
            "the bottleneck is not",
            "the slope is not",
            "the pattern is not",
            "the resource problem is not",
            "the efficiency issue is not",
            "the cost profile is not",
            "the systems pressure is not",
        ),
    ),
    PhraseFamily(
        label="context_efficiency_tradeoff",
        pattern=re.compile(r"\b(?:the\s+)?context[- ]efficiency tradeoff\b", re.I),
        variants=(
            "the long-context resource tradeoff",
            "the cost of extended context",
            "the efficiency cost of longer inputs",
            "the resource burden of longer prompts",
            "the systems cost of bigger windows",
            "the serving tradeoff for long context",
            "the runtime cost of extended context",
            "the memory and compute burden",
        ),
    ),
    PhraseFamily(
        label="this_figure_really",
        pattern=re.compile(r"\bthis figure really\b", re.I),
        variants=(
            "the chart",
            "the visual",
            "this panel",
            "the figure",
            "the plotted result",
            "the diagram",
            "this evidence",
            "the slide",
        ),
    ),
    PhraseFamily(
        label="long_context",
        pattern=re.compile(r"\b(?:the\s+)?long[- ]context\b", re.I),
        variants=(
            "extended context",
            "the long-context setting",
            "the larger context budget",
            "the extended input window",
            "the long-window setting",
            "the expanded context span",
            "the large-context regime",
            "the longer prompt window",
        ),
    ),
    PhraseFamily(
        label="context_budget",
        pattern=re.compile(r"\b(?:the\s+)?context budget\b", re.I),
        variants=(
            "the available context",
            "the token budget",
            "the input budget",
            "the prompt budget",
            "the usable context span",
            "the context allocation",
            "the window size",
            "the extended-input budget",
        ),
    ),
    PhraseFamily(
        label="large_context_regime",
        pattern=re.compile(r"\b(?:the\s+)?large[- ]context regime\b|\b(?:the\s+)?large context\b", re.I),
        variants=(
            "the extended-input setting",
            "the longer-window case",
            "the high-token setting",
            "the expanded-context setting",
            "the large-window case",
            "the long-prompt setting",
            "the wide-context case",
            "the extended-window regime",
        ),
    ),
    PhraseFamily(
        label="awkward_4k_128k",
        pattern=re.compile(r"\b4k to a 128k\b", re.I),
        variants=(
            "4,000 to 128,000 tokens",
            "a 32x context expansion",
            "the 4,000-to-128,000-token span",
            "the long-context sweep",
            "short-to-six-figure token windows",
            "the extended-context range",
        ),
    ),
    PhraseFamily(
        label="tokens_128000",
        pattern=re.compile(r"\b(?:to\s+)?128,?000 tokens\b", re.I),
        variants=(
            "six-figure token windows",
            "the 128K-token regime",
            "very long context budgets",
            "the extended-token range",
            "large context windows",
            "the high-context end of the sweep",
        ),
    ),
    PhraseFamily(
        label="flattening_curve",
        pattern=re.compile(r"\bflattening (?:that|this|the) curve\b", re.I),
        variants=(
            "reducing that slope",
            "lowering the cost profile",
            "easing that pressure",
            "making the resource pattern less steep",
            "improving the operating frontier",
            "bringing the bottleneck under control",
        ),
    ),
    PhraseFamily(
        label="budget_memory_explicitly",
        pattern=re.compile(
            r"\bsystems that budget memory explicitly\b|\bthat budget memory explicitly\b|\bbudget memory explicitly\b",
            re.I,
        ),
        variants=(
            "memory-budgeted systems",
            "designs that track memory cost directly",
            "architectures with explicit memory budgets",
            "systems planned around cache cost",
            "models that manage memory as a first-class constraint",
            "designs with memory cost built in",
        ),
    ),
    PhraseFamily(
        label="part_of_the_design",
        pattern=re.compile(r"\bpart of the design\b", re.I),
        variants=(
            "built into the method",
            "part of the mechanism",
            "handled by the system itself",
            "accounted for in the architecture",
            "folded into the training setup",
            "made explicit in the model design",
        ),
    ),
    PhraseFamily(
        label="architecture_itself",
        pattern=re.compile(r"\bthe architecture itself\b", re.I),
        variants=(
            "the model design",
            "the system design",
            "the architecture",
            "the method",
            "the design of the model",
            "the underlying system",
        ),
    ),
    PhraseFamily(
        label="this_figure_shows",
        pattern=re.compile(r"\bthis figure shows\b", re.I),
        variants=(
            "the plotted result shows",
            "the visual makes clear",
            "the chart shows",
            "the panel shows",
            "the evidence points to",
            "the figure makes visible",
        ),
    ),
)


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


def _is_figure_detail(slide_number: int, title: str) -> bool:
    return slide_number >= 100000 or "figure detail" in title.lower()


def _variant_for(family: PhraseFamily, *, key: str) -> str:
    digest = hashlib.sha256(f"{family.label}:{key}".encode("utf-8")).hexdigest()
    return family.variants[int(digest[:8], 16) % len(family.variants)]


def _match_case(variant: str, matched: str) -> str:
    return variant[:1].upper() + variant[1:] if matched[:1].isupper() else variant


def repair_text(text: str, *, key: str, stats: RepairStats) -> str:
    repaired = text
    for family in PHRASE_FAMILIES:
        def replace(match: re.Match[str]) -> str:
            assert stats.replacements_by_family is not None
            stats.replacements_by_family[family.label] = stats.replacements_by_family.get(family.label, 0) + 1
            match_key = f"{key}:{match.start()}:{match.group(0).lower()}"
            return _match_case(_variant_for(family, key=match_key), match.group(0))

        repaired = family.pattern.sub(replace, repaired)
    return repaired


def repair_markdown(path: Path, *, stats: RepairStats, dry_run: bool) -> None:
    original = path.read_text(encoding="utf-8")
    headers = list(SLIDE_HEADER_RE.finditer(original))
    if not headers:
        return

    pieces: list[str] = [original[: headers[0].start()]]
    turns_changed = 0
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(original)
        block = original[start:end]
        slide_number = int(match.group("number"))
        title = match.group("title").strip()
        if not _is_figure_detail(slide_number, title):
            pieces.append(block)
            continue

        lines: list[str] = []
        block_turns_changed = 0
        for line_number, line in enumerate(block.splitlines(), start=1):
            turn_match = TURN_LINE_RE.match(line)
            if turn_match:
                key = f"{path.parent.name}:{slide_number}:{turn_match.group('text')}"
                repaired = repair_text(turn_match.group("text"), key=key, stats=stats)
                if repaired != turn_match.group("text"):
                    turns_changed += 1
                    block_turns_changed += 1
                line = f"{turn_match.group('prefix')}{repaired}{turn_match.group('suffix')}"
            lines.append(line)
        repaired_block = "\n".join(lines)
        if not block_turns_changed:
            def repair_say(match: re.Match[str]) -> str:
                nonlocal turns_changed
                body = match.group("body")
                key = f"{path.parent.name}:{slide_number}:{body}"
                repaired = repair_text(body, key=key, stats=stats)
                if repaired != body:
                    turns_changed += 1
                return f"{match.group('label')}{repaired}"

            repaired_block = SAY_BLOCK_RE.sub(repair_say, repaired_block)
        if not repaired_block.endswith("\n"):
            repaired_block += "\n"
        pieces.append(repaired_block)

    repaired = "".join(pieces)
    stats.markdown_turns_changed += turns_changed
    if not dry_run and repaired != original:
        path.write_text(repaired, encoding="utf-8")


def repair_script_json(path: Path, *, script_md: Path, stats: RepairStats, dry_run: bool) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    turns_changed = 0

    for segment in data.get("segments", []):
        original = str(segment.get("text") or "")
        try:
            slide_number = int(segment.get("slide_number", 0) or 0)
        except (TypeError, ValueError):
            slide_number = 0
        title = str(segment.get("slide_title") or "")
        if not original or not _is_figure_detail(slide_number, title):
            continue
        key = f"{script_md.parent.name}:{slide_number}:{original}"
        repaired = repair_text(original, key=key, stats=stats)
        if repaired != original:
            segment["text"] = repaired
            turns_changed += 1

    slides = data.get("metadata", {}).get("slides", [])
    if isinstance(slides, list):
        for slide in slides:
            try:
                slide_number = int(slide.get("slide_number", 0) or 0)
            except (TypeError, ValueError):
                slide_number = 0
            title = str(slide.get("title") or "")
            if not _is_figure_detail(slide_number, title):
                continue
            for turn in slide.get("turns", []):
                original = str(turn.get("text") or "")
                if not original:
                    continue
                key = f"{script_md.parent.name}:{slide_number}:{original}"
                repaired = repair_text(original, key=key, stats=stats)
                if repaired != original:
                    turn["text"] = repaired
                    turns_changed += 1

    stats.json_turns_changed += turns_changed
    if not dry_run and turns_changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks-dir", type=Path)
    parser.add_argument("--script-md", type=Path, action="append", default=[])
    parser.add_argument("--script-name", default="6_presenter_script_notebooklm_2host.md")
    parser.add_argument("--script-json-dir", default="video_build_notebooklm")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _script_paths(args.decks_dir, args.script_md, script_name=args.script_name)
    stats = RepairStats(scripts_seen=len(paths))
    for script_md in paths:
        repair_markdown(script_md, stats=stats, dry_run=args.dry_run)
        repair_script_json(
            script_md.parent / args.script_json_dir / "script.json",
            script_md=script_md,
            stats=stats,
            dry_run=args.dry_run,
        )

    report = {"dry_run": args.dry_run, "stats": asdict(stats)}
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(
        "narration series fix: "
        f"{stats.markdown_turns_changed} markdown turns, "
        f"{stats.json_turns_changed} json turns",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
