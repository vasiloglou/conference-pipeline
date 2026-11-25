"""What a rendered screenshot is a picture OF.

The vision judge sees pixels and a sequence number. It has never been told the
slide's claim, the archetype, or the grounding kind, which is why it read the
compiler's mandatory honesty label as a stray production note and could not
distinguish a figure that omits numbers from one that is forbidden to carry
them.

This module does the join: screenshot filename to slide id to figure manifest.
Two naming schemes exist and both are supported -- the production route writes
`slide_003.png` and resolves the number through the render spec, while an
ad-hoc rasterisation of `assets/compiled` writes `slide-03.png` in sorted
figure order. A filename that carries the figure id is preferred over either,
because position is a guess and an id is not.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTEXT_VERSION = "figure-judge-context/1"
_NUMBER = re.compile(r"(\d+)")


def _manifests(compiled_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if not compiled_dir.is_dir():
        return found
    for path in sorted(compiled_dir.glob("*.manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            found.append(payload)
    return found


def _context(manifest: Mapping[str, Any], join: str) -> dict[str, Any]:
    grounding = manifest.get("grounding")
    kind = grounding.get("kind") if isinstance(grounding, dict) else ""
    return {
        "figure_id": manifest.get("figure_id", ""),
        "slide_id": manifest.get("slide_id", ""),
        "claim": manifest.get("claim", ""),
        "takeaway": manifest.get("takeaway", ""),
        "archetype": manifest.get("archetype", ""),
        "family": manifest.get("family", ""),
        "grounding_kind": kind or "",
        "glyphs_used": list(manifest.get("glyphs_used") or []),
        "join_method": join,
        "context_version": CONTEXT_VERSION,
    }


def build_figure_context(
    *,
    compiled_dir: Path,
    screenshots: Sequence[Path],
    render_spec: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Context for each screenshot, keyed by filename."""

    manifests = _manifests(compiled_dir)
    if not manifests:
        return {}
    by_slide = {str(m.get("slide_id") or ""): m for m in manifests}
    by_figure = {str(m.get("figure_id") or ""): m for m in manifests}

    slide_ids: dict[int, str] = {}
    slides = (render_spec or {}).get("slides")
    if isinstance(slides, list):
        for index, slide in enumerate(slides, start=1):
            if isinstance(slide, dict) and slide.get("slide_id"):
                slide_ids[index] = str(slide["slide_id"])

    out: dict[str, dict[str, Any]] = {}
    for screenshot in screenshots:
        name = Path(screenshot).name
        stem = Path(screenshot).stem

        # An id in the filename beats any positional guess.
        figure_hit = next((fid for fid in by_figure if fid and fid in stem), "")
        if figure_hit:
            out[name] = _context(by_figure[figure_hit], "figure_id")
            continue

        match = _NUMBER.search(stem)
        if not match:
            continue
        number = int(match.group(1))
        slide_id = slide_ids.get(number, "")
        if slide_id and slide_id in by_slide:
            out[name] = _context(by_slide[slide_id], "render_spec")
        elif 1 <= number <= len(manifests):
            # Last resort: the nth screenshot is the nth figure in sorted order.
            # Correct for a directory rasterised in one pass, and wrong the
            # moment a figure is added, so it says which rule it used.
            out[name] = _context(manifests[number - 1], "positional")
    return out


__all__ = ["CONTEXT_VERSION", "build_figure_context"]
