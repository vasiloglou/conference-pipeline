#!/usr/bin/env python3
"""Convert the delivery-final AI4Physics narrative into presenter-video input."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SLIDE_HEADER = re.compile(
    r"(?m)^## Slide\s+(?P<number>\d+)\s+—\s+(?P<title>.+?)\s+\(`(?P<slide_id>[^`]+)`\).*?$"
)
NAMED_BLOCK = r"(?ms)^\*\*{name}(?:\s*\([^)]*\))?:\*\*\s*(.*?)(?=^\*\*[^\n]+?:\*\*|^<!--|^## Slide\s|\Z)"


def _block(text: str, name: str) -> str:
    match = re.search(NAMED_BLOCK.format(name=re.escape(name)), text)
    return match.group(1).strip() if match else ""


def _spoken(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*>\s?", "", raw).strip()
        if line:
            lines.append(line)
    return "\n> ".join(lines)


def _backup_ids(slides_md: Path) -> set[str]:
    text = slides_md.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n", text)
    result: set[str] = set()
    for block in blocks:
        id_match = re.search(r"<!--\s*slide_id:\s*([^\s]+)\s*-->", block)
        if id_match and re.search(r"(?m)^narration_status:\s*backup\s*$", block):
            result.add(id_match.group(1))
    return result


def _zoom_counts(slides_html: Path | None) -> dict[int, int]:
    """Count figure-zoom companion sections per parent logical slide."""
    if slides_html is None:
        return {}
    text = slides_html.read_text(encoding="utf-8")
    counts: dict[int, int] = {}
    for tag in re.findall(r"<section\b[^>]*>", text):
        if "slide-figure-zoom" not in tag:
            continue
        parent_match = re.search(r'data-markdown-slide="(\d+)"', tag)
        if not parent_match:
            raise SystemExit(f"Zoom section without data-markdown-slide in {slides_html}")
        parent = int(parent_match.group(1))
        counts[parent] = counts.get(parent, 0) + 1
    return counts


def convert(narrative: Path, slides_md: Path, slides_html: Path | None = None) -> str:
    text = narrative.read_text(encoding="utf-8")
    headers = list(SLIDE_HEADER.finditer(text))
    backups = _backup_ids(slides_md)
    zoom_counts = _zoom_counts(slides_html)
    rendered: list[str] = [
        "# Presenter Script: AI for Physics Through NeurIPS, ICLR, and ICML",
        "",
        "Derived deterministically from the delivery-final `narrative.md`.",
        "",
    ]
    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[start:end]
        number = int(header.group("number"))
        title = header.group("title").strip()
        slide_id = header.group("slide_id")
        establishes = re.sub(r"\s+", " ", _block(body, "Establishes")).strip()
        script = _spoken(_block(body, "Script"))
        figure = re.sub(r"\s+", " ", _block(body, "Figure/enrichment leads")).strip()
        if slide_id in backups:
            script = (
                "These are the sources behind the preceding finding. "
                "We will leave them on screen briefly and keep moving."
            )
        next_title = headers[index + 1].group("title").strip() if index + 1 < len(headers) else ""
        zoom_count = zoom_counts.get(number, 0)
        rendered.extend(
            [
                f"### [SLIDE {number}] {title}",
                "",
                "**SAY:**",
                f"> {script}",
                "",
                "**CONTEXT:**",
                f"- Slide ID: {slide_id}",
                f"- {establishes}" if establishes else "- Delivery-final narrative slide.",
            ]
        )
        if figure:
            rendered.extend(["", "**FIGURE:**", figure])
        # When zoom companions follow, the spoken bridge moves to the last zoom
        # beat so the parent's dialogue doesn't announce the next topic and then
        # linger on a figure close-up.
        if next_title and not zoom_count:
            rendered.extend(["", "**TRANSITION:**", f"Next, we turn to {next_title}."])
        rendered.extend(["", "---", ""])

        # One [ZOOM N] beat per parent: the builder keys dialogue by slide number
        # and maps that single beat across every zoom frame that follows the slide.
        if zoom_count:
            zoom_figure = figure or f"Enlarged view of the figure on '{title}'."
            rendered.extend(
                [
                    f"### [ZOOM {number}] {title} — figure close-up",
                    "",
                    "**SAY:**",
                    f"> {zoom_figure}",
                    "",
                    "**CONTEXT:**",
                    f"- Slide ID: {slide_id}-zoom",
                    f"- Enlarged view of the figure on slide {number}; a quick beat, then move on.",
                    "",
                    "**FIGURE:**",
                    zoom_figure,
                ]
            )
            if next_title:
                rendered.extend(["", "**TRANSITION:**", f"Next, we turn to {next_title}."])
            rendered.extend(["", "---", ""])
    if len(headers) != 49:
        raise SystemExit(f"Expected 49 narrative slides, found {len(headers)}")
    unknown_parents = sorted(set(zoom_counts) - {int(h.group("number")) for h in headers})
    if unknown_parents:
        raise SystemExit(
            f"Zoom sections reference slides missing from the narrative: {unknown_parents}"
        )
    return "\n".join(rendered).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrative", type=Path, required=True)
    parser.add_argument("--slides-md", type=Path, required=True)
    parser.add_argument(
        "--slides-html",
        type=Path,
        help=(
            "Rendered Reveal.js deck; when given, figure-zoom companion sections "
            "become [ZOOM N] beats so video frame timing follows the narration."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        convert(args.narrative, args.slides_md, args.slides_html), encoding="utf-8"
    )
    print(f"Presenter script written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
