"""Build a ``key_points`` spec from a deck contract slide.

The operator's outcome "replace the figure with the slide's text layout"
(docs/operations/figure-fidelity-policy.md, item 9) needs a figure that says
exactly what the slide says and nothing else. This builds it mechanically:
the slide's title, the first sentence of its first claim (a spec's claim is
one sentence) and its visible-text lines, verbatim, with no takeaway (a
takeaway the slide does not state would be invented).

Every drawn string is then checked against the one grounding rule
(`figures.grounding`): it must be named by the slide outright. Verbatim text
always is, so the check passes trivially; it is here so that a future edit
which paraphrases, merges or adds a line fails loudly instead of shipping.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from conference_pipeline.figures.errors import FigureSpecError
from conference_pipeline.figures.grounding import STRONG, grounding_level, grounding_text
from conference_pipeline.figures.spec import SPEC_VERSION

TEXT_LAYOUT_ARCHETYPE = "key_points"
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def first_sentence(text: str) -> str:
    """The leading sentence of ``text``, verbatim; the whole text if it has one."""

    return _SENTENCE_END_RE.split(text.strip(), maxsplit=1)[0].strip()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).split()) for item in value if str(item or "").strip()]


def key_points_spec(slide: Mapping[str, Any], *, figure_id: str = "", section: str = "") -> dict[str, Any]:
    """A ``key_points`` spec carrying only ``slide``'s own words.

    Raises ``FigureSpecError`` when the slide has no visible text or no claim
    to typeset, or when any drawn string is not grounded in the slide.
    """

    slide_id = str(slide.get("slide_id") or "").strip()
    points = _strings(slide.get("visible_text"))
    if not points:
        raise FigureSpecError(f"{slide_id or 'slide'} has no visible_text to typeset")
    # Only fields the grounding pool reads (`grounding.grounding_text`), so the
    # check below is judged against the same text the figure is built from.
    claims = _strings(slide.get("claims"))
    if not claims:
        raise FigureSpecError(f"{slide_id or 'slide'} has no claim to typeset")
    title = (_strings(slide.get("title")) or [""])[0]
    payload: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "figure_id": figure_id or f"{slide_id}-figure",
        "slide_id": slide_id,
        "archetype": TEXT_LAYOUT_ARCHETYPE,
        "title": title,
        "claim": first_sentence(claims[0]),
        "takeaway": "",
        "section": section or str(slide.get("section_id") or ""),
        "grounding": {"kind": "schematic", "note": "slide text layout: visible_text and claim, verbatim"},
        "content": {"points": [{"text": point} for point in points]},
    }
    ungrounded = ungrounded_strings(payload, slide)
    if ungrounded:
        raise FigureSpecError(f"text layout would draw strings the slide does not state: {ungrounded}")
    return payload


def drawn_strings(payload: Mapping[str, Any]) -> list[str]:
    """The title, claim, takeaway and points a ``key_points`` spec draws."""

    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    points = [str(point.get("text") or "") for point in content.get("points") or [] if isinstance(point, Mapping)]
    head = [str(payload.get(key) or "") for key in ("title", "claim", "takeaway")]
    return [text for text in (*head, *points) if text.strip()]


def ungrounded_strings(payload: Mapping[str, Any], slide: Mapping[str, Any]) -> list[str]:
    """Drawn strings the slide does not name outright (grounding level below STRONG)."""

    pool = grounding_text(slide)
    return [text for text in drawn_strings(payload) if grounding_level(text, pool)[0] != STRONG]


__all__ = ["TEXT_LAYOUT_ARCHETYPE", "drawn_strings", "first_sentence", "key_points_spec", "ungrounded_strings"]
