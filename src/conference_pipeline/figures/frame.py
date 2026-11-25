"""The figure frame: title, claim strip, plot area, takeaway and badge.

Every archetype draws into the same frame, so compiled figures across a deck
share a baseline grid, and each layout only has to fill ``plot_area``.
"""

from __future__ import annotations

from dataclasses import dataclass

from conference_pipeline.figures import text as textlib
from conference_pipeline.figures import tokens
from conference_pipeline.figures.geometry import Box
from conference_pipeline.figures.spec import (
    FAMILY_QUANTITATIVE,
    FAMILY_SCHEMATIC,
    FigureSpec,
)
from conference_pipeline.figures.svgdoc import SvgDoc

TITLE_GAP = 12.0
CLAIM_GAP = 18.0
TAKEAWAY_GAP = 16.0
BADGE_GAP = 10.0


@dataclass(frozen=True)
class Frame:
    """Where the chrome went and what is left for the drawing."""

    plot: Box
    header_bottom: float
    takeaway_top: float


# Families whose figures could be mistaken for measured data if unlabelled: a
# curve or a chart-shaped figure needs the badge, a box-and-arrow mechanism or a
# feature matrix does not, and carrying it there only adds noise to every slide.
BADGED_FAMILIES = frozenset({FAMILY_SCHEMATIC, FAMILY_QUANTITATIVE})


def _badge_text(spec: FigureSpec) -> str:
    if spec.family not in BADGED_FAMILIES:
        return ""
    if spec.grounding.kind == "schematic":
        return tokens.SCHEMATIC_BADGE_TEXT
    if spec.grounding.kind == "hybrid":
        return tokens.HYBRID_BADGE_TEXT
    return ""


def draw_frame(doc: SvgDoc, spec: FigureSpec) -> Frame:
    """Draw title, claim, takeaway and grounding badge; return the plot area."""

    left = tokens.MARGIN
    right = doc.width - tokens.MARGIN
    width = right - left
    cursor = tokens.MARGIN

    title = spec.title.strip() or ""
    if title:
        block = textlib.fit(title, max_width=width, max_lines=2, size=tokens.FONT_TITLE,
                            min_size=tokens.FONT_CLAIM, bold=True)
        box = doc.text_block(block, left, cursor, fill=tokens.TEXT)
        cursor = box.y1 + TITLE_GAP

    claim = spec.claim.strip()
    if claim:
        block = textlib.fit(claim, max_width=width, max_lines=2, size=tokens.FONT_CLAIM,
                            min_size=tokens.FONT_FLOOR)
        box = doc.text_block(block, left, cursor, fill=tokens.ACCENT)
        cursor = box.y1 + CLAIM_GAP

    if spec.synthesis_label:
        block = textlib.fit(spec.synthesis_label, max_width=width, max_lines=1,
                            size=tokens.FONT_BADGE, min_size=tokens.FONT_BADGE)
        box = doc.text_block(block, left, cursor, fill=tokens.TEXT_MUTED)
        cursor = box.y1 + TITLE_GAP

    header_bottom = cursor

    # Bottom band: the takeaway, and the grounding badge to its right.
    bottom = doc.height - tokens.MARGIN
    badge = _badge_text(spec)
    badge_width = textlib.measure(badge, tokens.FONT_BADGE) if badge else 0.0

    takeaway = spec.takeaway.strip()
    takeaway_top = bottom
    if takeaway:
        available = width - (badge_width + 40.0 if badge else 0.0)
        block = textlib.fit(takeaway, max_width=available, max_lines=1,
                            size=tokens.FONT_TAKEAWAY, min_size=tokens.FONT_FLOOR)
        height = block.height
        takeaway_top = bottom - height
        doc.text_block(block, left, takeaway_top, fill=tokens.EMPHASIS)
    if badge:
        baseline = bottom - tokens.FONT_BADGE * 0.22
        doc.text(badge, right, baseline, size=tokens.FONT_BADGE, fill=tokens.TEXT_MUTED, anchor="end")
        takeaway_top = min(takeaway_top, baseline - tokens.FONT_BADGE * 0.78)

    plot_top = header_bottom
    plot_bottom = takeaway_top - TAKEAWAY_GAP if (takeaway or badge) else bottom
    plot = Box(left, plot_top, width, max(60.0, plot_bottom - plot_top))
    return Frame(plot=plot, header_bottom=header_bottom, takeaway_top=takeaway_top)
