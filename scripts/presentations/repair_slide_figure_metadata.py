#!/usr/bin/env python3
"""Repair slide FIGURE_NEED and image alt text from FigureSpec metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SLIDE_RE = re.compile(
    r"(?ms)^## Slide (?P<num>\d+): (?P<title>[^\n]+)\n(?P<body>.*?)(?=^## Slide \d+:|\Z)"
)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)]+)\)")
FIGURE_NEED_RE = re.compile(r"(?m)^FIGURE_NEED:\s*.*$")


def _clean(text: object, *, max_len: int = 170) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip(" ,;") + "."


def _spec_slide_number(row: dict[str, Any]) -> int | None:
    for source in (row.get("figure_spec"), row):
        if not isinstance(source, dict):
            continue
        try:
            return int(source.get("slide_number"))
        except (TypeError, ValueError):
            continue
    return None


def _load_specs(asset_inventory: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(asset_inventory.read_text(encoding="utf-8"))
    rows: list[Any] = []
    if isinstance(payload, dict):
        rows = payload.get("figure_specs") or payload.get("figures") or []
    elif isinstance(payload, list):
        rows = payload
    specs: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        spec = row.get("figure_spec") if isinstance(row.get("figure_spec"), dict) else row
        if not isinstance(spec, dict):
            continue
        slide_number = _spec_slide_number(row)
        if slide_number is not None:
            specs[slide_number] = spec
    return specs


def _role_from_spec(spec: dict[str, Any]) -> str:
    depiction = _clean(spec.get("depiction_type"), max_len=60).replace("_", " ")
    figure_need = _clean(spec.get("figure_need"), max_len=100).lower()
    semantic_goal = _clean(spec.get("semantic_goal"), max_len=100).lower()
    source = f"{depiction} {figure_need} {semantic_goal}".lower()
    if any(term in source for term in ("attack", "risk", "vulnerability", "failure")):
        return "failure mode evidence map"
    if any(term in source for term in ("architecture", "system", "module", "agent")):
        return "system architecture"
    if any(term in source for term in ("benchmark", "metric", "evaluation")):
        return "benchmark comparison"
    if any(term in source for term in ("workflow", "process", "loop", "flow")):
        return "mechanism workflow"
    if any(term in source for term in ("experiment", "validation", "ablation")):
        return "experiment workflow"
    return depiction or "mechanism evidence map"


def _figure_need(spec: dict[str, Any]) -> str:
    role = _role_from_spec(spec)
    goal = _clean(spec.get("semantic_goal"), max_len=150)
    entities = spec.get("entities") if isinstance(spec.get("entities"), list) else []
    entity_text = ", ".join(_clean(entity, max_len=40) for entity in entities[:3] if _clean(entity))
    if goal and entity_text:
        return f"{role.title()} showing {entity_text}: {goal}"
    if goal:
        return f"{role.title()} showing {goal}"
    original = _clean(spec.get("figure_need"), max_len=150)
    return f"{role.title()} showing {original}" if original else role.title()


def _alt_text(spec: dict[str, Any]) -> str:
    role = _role_from_spec(spec)
    goal = _clean(spec.get("semantic_goal"), max_len=120)
    entities = spec.get("entities") if isinstance(spec.get("entities"), list) else []
    entity_text = ", ".join(_clean(entity, max_len=32) for entity in entities[:3] if _clean(entity))
    if entity_text and goal:
        return f"{role.title()}: {entity_text} - {goal}"
    if goal:
        return f"{role.title()}: {goal}"
    return _figure_need(spec)


def repair_markdown(slides_md: Path, *, asset_inventory: Path) -> dict[str, Any]:
    text = slides_md.read_text(encoding="utf-8")
    specs = _load_specs(asset_inventory)
    figure_need_updates = 0
    alt_updates = 0

    def replace_slide(match: re.Match[str]) -> str:
        nonlocal figure_need_updates, alt_updates
        num = int(match.group("num"))
        spec = specs.get(num)
        if not spec:
            return match.group(0)
        body = match.group("body")
        new_need = f"FIGURE_NEED: {_figure_need(spec)}"
        if FIGURE_NEED_RE.search(body):
            body, count = FIGURE_NEED_RE.subn(new_need, body, count=1)
            figure_need_updates += count
        image_alt = _alt_text(spec)

        def replace_image(image: re.Match[str]) -> str:
            nonlocal alt_updates
            alt_updates += 1
            return f"![{image_alt}]({image.group('target')})"

        body = IMAGE_RE.sub(replace_image, body, count=1)
        return f"## Slide {num}: {match.group('title')}\n{body}"

    repaired = SLIDE_RE.sub(replace_slide, text)
    if repaired != text:
        slides_md.write_text(repaired, encoding="utf-8")
    return {
        "slides_md": str(slides_md),
        "asset_inventory": str(asset_inventory),
        "figure_specs_loaded": len(specs),
        "figure_need_updates": figure_need_updates,
        "alt_updates": alt_updates,
        "changed": repaired != text,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides-md", type=Path, required=True)
    parser.add_argument("--asset-inventory", type=Path, required=True)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = repair_markdown(args.slides_md, asset_inventory=args.asset_inventory)
    payload = json.dumps(report, indent=2) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    print(
        "slide figure metadata repair: "
        f"{report['alt_updates']} alt updates, {report['figure_need_updates']} figure need updates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
